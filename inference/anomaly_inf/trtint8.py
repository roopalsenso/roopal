import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import json
import os

# ================= CONFIG =================
ENGINE_PATH = "/root/old-data/home/roopal/engines/anomaly/anomalyint8_backbone.trt"
MEMORY_BANK_PATH = "/root/old-data/home/roopal/models/onnx/patchcore_memory_bank_trt.npy"
METADATA_PATH = "/root/old-data/home/roopal/models/onnx/patchcore_metadata.json"

IMAGE_PATH = "/root/old-data/home/roopal/datasets/MvTecAD/MvTecAD/OnlyOkbottles_test/images/broken_large/004.png"
SAVE_DIR = "/root/old-data/home/roopal/inference/results/anomaly/int8trt_calib"

INPUT_H = 224
INPUT_W = 224
K = 9
# ==========================================

os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- Load memory bank ----------
memory_bank = np.load(MEMORY_BANK_PATH).astype(np.float32)

# L2 normalize memory bank
memory_bank = memory_bank / (np.linalg.norm(memory_bank, axis=1, keepdims=True) + 1e-8)
# ---------- Load metadata ----------
with open(METADATA_PATH) as f:
    meta = json.load(f)

print(meta)

image_threshold = meta["image_threshold"]
pixel_threshold = meta["pixel_threshold"]

image_min = meta["image_min"]
image_max = meta["image_max"]

pixel_min = meta["pixel_min"]
pixel_max = meta["pixel_max"]

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

# ---------- Image preprocessing ----------
img = cv2.imread(IMAGE_PATH)
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

# ---------- Allocate GPU memory ----------
d_input = cuda.mem_alloc(img.nbytes)

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

bindings = [int(d_input), int(d_out1), int(d_out2)]

# ---------- Run inference ----------
cuda.memcpy_htod(d_input,img)
context.execute_v2(bindings)

cuda.memcpy_dtoh(feat2,d_out1)
cuda.memcpy_dtoh(feat3,d_out2)

feat2 = feat2.astype(np.float32)
feat3 = feat3.astype(np.float32)

print("feat2 shape:", feat2.shape)
print("feat3 shape:", feat3.shape)

# Remove batch
feat2 = feat2[0]   # [512,28,28]
feat3 = feat3[0]   # [1024,14,14]

# ---------- Upsample layer3 ----------
upsampled = []

for c in range(feat3.shape[0]):
    channel = feat3[c]
    channel_up = cv2.resize(channel,(28,28),interpolation=cv2.INTER_LINEAR)
    upsampled.append(channel_up)

feat3 = np.stack(upsampled,axis=0)   # [1024,28,28]

# ---------- Concatenate ----------
embedding = np.concatenate([feat2,feat3],axis=0)  # [1536,28,28]

# ---------- Patch reshape ----------
# ---------- Patch reshape ----------
embedding = embedding.reshape(1536,-1).T

# L2 normalize embeddings (CRITICAL for PatchCore)
embedding = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8)
print("Embedding shape:",embedding.shape)
print("Memory bank shape:",memory_bank.shape)

# ---------- Euclidean KNN distance (MATCHES TRAINING) ----------
emb_norm = np.sum(embedding**2, axis=1, keepdims=True)
mem_norm = np.sum(memory_bank**2, axis=1)

dists = emb_norm + mem_norm - 2 * np.dot(embedding, memory_bank.T)
dists = np.maximum(dists,0)
dists = np.sqrt(dists)

# ---------- KNN ----------
knn = np.partition(dists,K,axis=1)[:,:K]
patch_scores = np.mean(knn,axis=1)
print("Patch score range:", patch_scores.min(), patch_scores.max())

# ---------- Reshape anomaly map ----------
anomaly_map = patch_scores.reshape(28,28)

print("Anomaly map stats")
print("min:", anomaly_map.min())
print("max:", anomaly_map.max())
print("mean:", anomaly_map.mean())

# ---------- Upsample ----------
anomaly_map = cv2.resize(anomaly_map,(224,224))

# ---------- Normalize anomaly map for visualization ----------
vis_map = anomaly_map / anomaly_map.max()
vis_map = np.clip(vis_map, 0, 1)

# ---------- Threshold ----------
mask = vis_map > 0.5
mask = (mask*255).astype(np.uint8)

# ---------- Heatmap ----------
vis_map = anomaly_map / anomaly_map.max()
vis_map = np.clip(vis_map,0,1)

heatmap = cv2.applyColorMap((vis_map*255).astype(np.uint8),
                            cv2.COLORMAP_JET)

overlay = cv2.addWeighted(orig,0.6,heatmap,0.4,0)
# ---------- Save results ----------
base = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

cv2.imwrite(os.path.join(SAVE_DIR,f"{base}_anomaly_map.jpg"),
            (anomaly_map*255).astype(np.uint8))

cv2.imwrite(os.path.join(SAVE_DIR,f"{base}_mask.jpg"),mask)

cv2.imwrite(os.path.join(SAVE_DIR,f"{base}_overlay.jpg"),overlay)

print("Saved results to:",SAVE_DIR)

print("Image anomaly score:", anomaly_map.max())
print("Raw max score:", patch_scores.max())
print("Normalized max score:", anomaly_map.max())