#!/usr/bin/env python
# coding: utf-8

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
from pathlib import Path
import csv
import os

# ==========================
# PERFORMANCE MEASUREMENT
gpu_times = []
full_times = []
# ==========================

# ==========================
ENGINE_PATH = "/root/old-data/home/roopal/engines/anomaly/anomalyfp32.trt"
DATASET_DIR = "/root/old-data/home/roopal/datasets/MvTecAD/MvTecAD/OnlyOkbottles_test/images"
RESULT_DIR = "/root/old-data/home/roopal/inference/results/anomaly"
INPUT_H, INPUT_W = 256, 256
THRESHOLD = 0.5

os.makedirs(RESULT_DIR, exist_ok=True)

# ==========================
# LOAD TRT ENGINE
logger = trt.Logger(trt.Logger.INFO)
runtime = trt.Runtime(logger)

with open(ENGINE_PATH, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# ==========================
# INPUT/OUTPUT TENSOR INFO
input_name = "inpt.1"

# Allocate memory for input
input_shape = tuple(engine.get_tensor_shape(input_name))
d_input = cuda.mem_alloc(int(np.prod(input_shape) * 4))

# Allocate memory for outputs
output_pred_score = np.empty((1,), dtype=np.float32)
output_anomaly_map = np.empty((1,1,224,224), dtype=np.float32)

dummy_bool1 = np.empty((1,), dtype=np.bool_)
dummy_bool2 = np.empty((1,1,224,224), dtype=np.bool_)

d_pred_score = cuda.mem_alloc(output_pred_score.nbytes)
d_anomaly_map = cuda.mem_alloc(output_anomaly_map.nbytes)
d_bool1 = cuda.mem_alloc(dummy_bool1.nbytes)
d_bool2 = cuda.mem_alloc(dummy_bool2.nbytes)

bindings = [
    int(d_input),
    int(d_pred_score),
    int(d_bool1),
    int(d_anomaly_map),
    int(d_bool2)
]

# ==========================
# PREPROCESS FUNCTION
def preprocess(img_path):

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_W, INPUT_H))

    img = img.astype(np.float32) / 255.0

    img = np.transpose(img, (2,0,1))[np.newaxis,:,:,:]
    img = np.ascontiguousarray(img)

    return img

# ==========================
# INFERENCE FUNCTION
def run_fp32_trt(img_path):

    img_input = preprocess(img_path)

    # -------- FULL PIPELINE TIMER --------
    start_full = cuda.Event()
    end_full = cuda.Event()

    start_full.record()

    cuda.memcpy_htod(d_input, img_input)

    # -------- GPU TIMER --------
    start_gpu = cuda.Event()
    end_gpu = cuda.Event()

    start_gpu.record()
    context.execute_v2(bindings)
    end_gpu.record()
    end_gpu.synchronize()

    cuda.memcpy_dtoh(output_pred_score, d_pred_score)

    end_full.record()
    end_full.synchronize()

    # save times
    gpu_times.append(start_gpu.time_till(end_gpu))
    full_times.append(start_full.time_till(end_full))

    return float(output_pred_score[0])

# ==========================
# RUN ON DATASET FOLDERS

test_root = Path(DATASET_DIR)

results = []

for class_dir in test_root.iterdir():

    if class_dir.is_dir():

        for image_path in class_dir.glob("*.png"):

            score = run_fp32_trt(str(image_path))

            pred_label = score > THRESHOLD

            print(f"[{class_dir.name}/{image_path.name}] → Anomaly Score: {score:.4f}, Pred Label: {pred_label}")

            results.append([
                class_dir.name,
                image_path.name,
                f"{score:.4f}",
                pred_label
            ])

# ==========================
# SAVE CSV

csv_path = os.path.join(RESULT_DIR, "trt_fp32_scores.csv")

with open(csv_path, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Image Folder",
        "Image Name",
        "Anomaly Score",
        "Predicted Label"
    ])

    writer.writerows(results)

print(f"\nSaved TRT FP32 results to {csv_path}")

# ==========================
# PERFORMANCE SUMMARY
# ==========================

gpu_times = np.array(gpu_times)
full_times = np.array(full_times)

print("\n===== PERFORMANCE SUMMARY =====")
print(f"GPU Compute Time  : {gpu_times.mean():.2f} ms")
print(f"Full Pipeline Time (H2D + Compute + D2H): {full_times.mean():.2f} ms")
print(f"FPS (compute only): {1000.0 / gpu_times.mean():.2f}")
print("================================\n")