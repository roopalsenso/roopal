import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import json
import os
import csv
from pathlib import Path

# ==========================
# PERFORMANCE MEASUREMENT
gpu_times = []
full_times = []
# ==========================

# ================= CONFIG =================
ENGINE_PATH = "/root/old-data/home/roopal/engines/anomaly/anomalyint8_backbone.trt"
MEMORY_BANK_PATH = "/root/old-data/home/roopal/models/onnx/patchcore_memory_bank_trt.npy"
METADATA_PATH = "/root/old-data/home/roopal/models/onnx/patchcore_metadata.json"

DATASET_DIR = "/root/old-data/home/roopal/datasets/MvTecAD/MvTecAD/OnlyOkbottles_test/images"

SAVE_DIR = "/root/old-data/home/roopal/inference/results/anomaly/int8trt_calib"
CSV_DIR = "/root/old-data/home/roopal/inference/results/anomaly"

INPUT_H = 224
INPUT_W = 224
K = 9
NUM_IMAGES = 20
# ==========================================

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

scores = []

# ---------- Load memory bank ----------
memory_bank = np.load(MEMORY_BANK_PATH).astype(np.float32)
memory_bank = memory_bank / (np.linalg.norm(memory_bank, axis=1, keepdims=True) + 1e-8)

# ---------- Load metadata ----------
with open(METADATA_PATH) as f:
    meta = json.load(f)

print(meta)

# ---------- TensorRT setup ----------
logger = trt.Logger(trt.Logger.INFO)
runtime = trt.Runtime(logger)

with open(ENGINE_PATH, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

input_name = engine.get_tensor_name(0)
out1_name = engine.get_tensor_name(1)
out2_name = engine.get_tensor_name(2)

shape1 = tuple(engine.get_tensor_shape(out1_name))
shape2 = tuple(engine.get_tensor_shape(out2_name))

print("Output1:", shape1)
print("Output2:", shape2)

# ---------- Allocate GPU memory ----------
dtype_map = {
    trt.DataType.FLOAT: np.float32,
    trt.DataType.HALF:  np.float16,
    trt.DataType.INT8:  np.int8,
    trt.DataType.INT32: np.int32
}

out1_dtype = dtype_map[engine.get_tensor_dtype(out1_name)]
out2_dtype = dtype_map[engine.get_tensor_dtype(out2_name)]

feat2 = np.empty(shape1, dtype=out1_dtype)
feat3 = np.empty(shape2, dtype=out2_dtype)

d_out1 = cuda.mem_alloc(feat2.nbytes)
d_out2 = cuda.mem_alloc(feat3.nbytes)

# -------------------------------
# Dataset root
# -------------------------------
test_root = Path(DATASET_DIR)

image_count = 0

# ===============================
# INFERENCE LOOP
# ===============================

for class_dir in test_root.iterdir():

    if class_dir.is_dir():

        for img_path in class_dir.glob("*.png"):

            if image_count >= NUM_IMAGES:
                break

            img = cv2.imread(str(img_path))
            orig = cv2.resize(img.copy(), (224,224))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256,256))

            start = (256-224)//2
            img = img[start:start+224, start:start+224]

            img = img.astype(np.float32)/255.0

            mean = np.array([0.485,0.456,0.406])
            std  = np.array([0.229,0.224,0.225])

            img = (img - mean)/std

            img = np.transpose(img,(2,0,1))
            img = np.expand_dims(img,0)
            img = np.ascontiguousarray(img).astype(np.float32)

                        # GPU input
            d_input = cuda.mem_alloc(img.nbytes)
            bindings = [int(d_input), int(d_out1), int(d_out2)]

            # -------- FULL PIPELINE TIMER --------
            start_full = cuda.Event()
            end_full = cuda.Event()

            start_full.record()

            cuda.memcpy_htod(d_input, img)

            # -------- GPU COMPUTE TIMER --------
            start_gpu = cuda.Event()
            end_gpu = cuda.Event()

            start_gpu.record()
            context.execute_v2(bindings)
            end_gpu.record()
            end_gpu.synchronize()

            cuda.memcpy_dtoh(feat2, d_out1)
            cuda.memcpy_dtoh(feat3, d_out2)

            end_full.record()
            end_full.synchronize()

            # Save timings
            gpu_times.append(start_gpu.time_till(end_gpu))
            full_times.append(start_full.time_till(end_full))

            feat2_np = feat2.astype(np.float32)[0]
            feat3_np = feat3.astype(np.float32)[0]

            # ---------- Upsample ----------
            upsampled = []

            for c in range(feat3_np.shape[0]):
                channel = feat3_np[c]
                channel_up = cv2.resize(channel,(28,28),interpolation=cv2.INTER_LINEAR)
                upsampled.append(channel_up)

            feat3_np = np.stack(upsampled,axis=0)

            # ---------- Concatenate ----------
            embedding = np.concatenate([feat2_np,feat3_np],axis=0)

            embedding = embedding.reshape(1536,-1).T
            embedding = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8)

            # ---------- Distance ----------
            emb_norm = np.sum(embedding**2, axis=1, keepdims=True)
            mem_norm = np.sum(memory_bank**2, axis=1)

            dists = emb_norm + mem_norm - 2 * np.dot(embedding, memory_bank.T)
            dists = np.maximum(dists,0)
            dists = np.sqrt(dists)

            knn = np.partition(dists,K,axis=1)[:,:K]
            patch_scores = np.mean(knn,axis=1)

            anomaly_map = patch_scores.reshape(28,28)
            anomaly_map = cv2.resize(anomaly_map,(224,224))

            vis_map = anomaly_map / anomaly_map.max()
            vis_map = np.clip(vis_map,0,1)

            mask = (vis_map > 0.5).astype(np.uint8)*255

            heatmap = cv2.applyColorMap((vis_map*255).astype(np.uint8),
                                        cv2.COLORMAP_JET)

            overlay = cv2.addWeighted(orig,0.6,heatmap,0.4,0)

            base = img_path.stem

            cv2.imwrite(os.path.join(SAVE_DIR,f"{base}_anomaly_map.jpg"),
                        (anomaly_map*255).astype(np.uint8))

            cv2.imwrite(os.path.join(SAVE_DIR,f"{base}_mask.jpg"),mask)

            cv2.imwrite(os.path.join(SAVE_DIR,f"{base}_overlay.jpg"),overlay)

            score = patch_scores.max()

            pred_label = score > 0.5

            scores.append((class_dir.name, img_path.name, score, pred_label))

            print(f"[{class_dir.name}/{img_path.name}] → Anomaly Score: {score:.4f}, Pred Label: {pred_label}")

            image_count += 1

        if image_count >= NUM_IMAGES:
            break


# ===============================
# SAVE CSV
# ===============================

csv_path = os.path.join(CSV_DIR, "trtint8_scores.csv")

with open(csv_path,"w",newline="") as f:
    writer = csv.writer(f)

    writer.writerow(["Image Folder","Image Name","Anomaly Score","Predicted Label"])

    for row in scores:
        writer.writerow([row[0], row[1], f"{row[2]:.4f}", row[3]])

print(f"\nSaved scores for {len(scores)} images to {csv_path}")

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