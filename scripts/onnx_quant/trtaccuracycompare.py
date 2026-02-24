# trtaccuracycompare.py
import os
import glob
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# ---------------- CONFIG ----------------
trt_engines = {
    "FP32": "/root/old-data/home/roopal/engines/semsons512fp32.trt",
    "FP16": "/root/old-data/home/roopal/engines/semsons512fp16.trt",
    "INT8": "/root/old-data/home/roopal/engines/semsons512int8.trt",
}

dataset_images = "/root/old-data/home/roopal/datasets/test_offline_dataset/images/*.jpg"
dataset_labels = "/root/old-data/home/roopal/datasets/test_offline_dataset/annotations/*.png"

num_classes = 2  # class 0 = background, class 1 = foreground

# ---------------- HELPER FUNCTIONS ----------------
def compute_class_iou(pred, target, cls):
    pred_inds = (pred == cls)
    target_inds = (target == cls)
    intersection = np.logical_and(pred_inds, target_inds).sum()
    union = np.logical_or(pred_inds, target_inds).sum()
    return intersection / union if union != 0 else 1.0

def compute_mean_iou(pred, target, num_classes):
    return np.mean([compute_class_iou(pred, target, cls) for cls in range(num_classes)])

# ---------------- LOAD DATA FILES ----------------
image_files = sorted(glob.glob(dataset_images))
label_files = sorted(glob.glob(dataset_labels))
assert len(image_files) == len(label_files), "Mismatch between images and labels"
print(f"Total images found: {len(image_files)}\n")

# ---------------- TRT LOGGER ----------------
trt_logger = trt.Logger(trt.Logger.INFO)

# ---------------- STORE RESULTS ----------------
results = {}

# ---------------- LOOP OVER TRT ENGINES ----------------
for quant_type, engine_path in trt_engines.items():
    print(f"Running engine: {quant_type} -> {engine_path}")

    with open(engine_path, "rb") as f:
        engine_data = f.read()

    runtime = trt.Runtime(trt_logger)
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()

    # Get input/output shapes
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    input_shape = tuple(engine.get_tensor_shape(input_name))
    output_shape = tuple(engine.get_tensor_shape(output_name))
    output_dtype = np.float32  # We'll cast INT8 outputs to float32 for comparison

    total_iou = total_bg_iou = total_fg_iou = 0

    for img_path, lbl_path in zip(image_files, label_files):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (input_shape[3], input_shape[2])).astype(np.float32)/255.0
        img_input = np.transpose(img_resized, (2,0,1))[None,...].astype(np.float32)
        img_input = np.ascontiguousarray(img_input)

        # Allocate GPU memory
        d_input = cuda.mem_alloc(int(img_input.nbytes))
        d_output = cuda.mem_alloc(int(np.prod(output_shape) * np.dtype(output_dtype).itemsize))
        cuda.memcpy_htod(d_input, img_input)

        # Run inference
        context.execute_v2([int(d_input), int(d_output)])

        # Copy output back
        output_trt = np.empty(output_shape, dtype=output_dtype)
        cuda.memcpy_dtoh(output_trt, d_output)

        # Convert to 0/1 mask
        pred = output_trt[0]
        if len(pred.shape) == 3 and pred.shape[0] == num_classes:
            pred = np.argmax(pred, axis=0)

        target = cv2.imread(lbl_path, 0)
        target_resized = cv2.resize(target, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)

        total_iou += compute_mean_iou(pred, target_resized, num_classes)
        total_bg_iou += compute_class_iou(pred, target_resized, 0)
        total_fg_iou += compute_class_iou(pred, target_resized, 1)

    avg_iou = total_iou / len(image_files)
    avg_bg_iou = total_bg_iou / len(image_files)
    avg_fg_iou = total_fg_iou / len(image_files)
    results[quant_type] = avg_iou

    print(f"{quant_type} Results:")
    print(f"  Background IoU: {avg_bg_iou:.4f}")
    print(f"  Foreground IoU: {avg_fg_iou:.4f}")
    print(f"  Mean IoU: {avg_iou:.4f}\n")

# ---------------- ACCURACY DROP COMPARISON ----------------
baseline_iou = results["FP32"]
print("Accuracy Drop vs FP32:")
for quant_type, mean_iou in results.items():
    if quant_type != "FP32":
        drop = (baseline_iou - mean_iou) / baseline_iou * 100
        print(f"  {quant_type}: {drop:.2f}%")