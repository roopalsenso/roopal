import os
import numpy as np
import torch
import cv2
import tensorrt as trt
import torch.nn.functional as F

# ================= CONFIG =================
TRT_LOGGER = trt.Logger()
ENGINE_PATH = "/root/old-data/home/roopal/engines/anomaly/anomalyint8_backbone.trt"
IMAGE_PATH = "/root/old-data/home/roopal/datasets/MvTecAD/MvTecAD/OnlyOkbottles_test/images/contamination/006.png"
SAVE_DIR = "/root/old-data/home/roopal/inference/results/int8trt_calib"
PATCHCORE_BANK = "/root/old-data/home/roopal/models/onnx/patchcore_memory_bank.npy"
PATCHCORE_METADATA = "/root/old-data/home/roopal/models/onnx/patchcore_metadata.json"

import numpy as np
import json

# Load memory bank
memory_bank = np.load(PATCHCORE_BANK)

# Load metadata
with open(PATCHCORE_METADATA, "r") as f:
    metadata = json.load(f)

INPUT_H, INPUT_W = 224, 224  # matches engine input
os.makedirs(SAVE_DIR, exist_ok=True)

# ================= LOAD ENGINE =================
with open(ENGINE_PATH, "rb") as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

num_tensors = engine.num_io_tensors

tensor_names = [engine.get_tensor_name(i) for i in range(num_tensors)]

# Map input/output
input_name = [n for n in tensor_names if 'input' in n][0]
output_names = [n for n in tensor_names if n != input_name]

# ================= LOAD PATCHCORE =================
memory_bank = np.load(PATCHCORE_BANK)
#metadata = np.load(PATCHCORE_METADATA, allow_pickle=True).item()

image_threshold = metadata["image_threshold"]
pixel_threshold = metadata["pixel_threshold"]
normalize = metadata["normalize"]

# ================= IMAGE PREPROCESS =================
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (INPUT_W, INPUT_H))
img_float = img.astype(np.float32) / 255.0

# CHW layout
input_tensor = np.transpose(img_float, (2, 0, 1)).ravel()

# ================= ALLOCATE CUDA MEMORY =================
import pycuda.driver as cuda
import pycuda.autoinit

d_input = cuda.mem_alloc(input_tensor.nbytes)

# compute output sizes
output_shapes = [engine.get_tensor_shape(n) for n in output_names]
output_sizes = [np.prod(s) for s in output_shapes]
d_outputs = [cuda.mem_alloc(s * 4) for s in output_sizes]  # float32

stream = cuda.Stream()

# ================= COPY INPUT =================
cuda.memcpy_htod_async(d_input, input_tensor, stream)

# ================= RUN INFERENCE =================
bindings = [int(d_input)] + [int(d) for d in d_outputs]
context.execute_async_v2(bindings, stream.handle, None)
stream.synchronize()

# ================= GET OUTPUT =================
outputs = []
for i, name in enumerate(output_names):
    out = np.empty(output_shapes[i], dtype=np.float32)
    cuda.memcpy_dtoh(out, d_outputs[i])
    outputs.append(torch.from_numpy(out))

layer2, layer3 = outputs

# Upsample layer3 to layer2
layer3_up = F.interpolate(layer3.unsqueeze(0), size=layer2.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)

# Concatenate along channel
features_concat = torch.cat([layer2, layer3_up], dim=1)
features_flat = features_concat.flatten(1)  # flatten spatial dims

# Compute distances to PatchCore memory bank
bank = torch.from_numpy(memory_bank)
distances = torch.cdist(features_flat.T, bank.T)  # [N_features, N_memory]
anomaly_map = distances.min(dim=1).values.reshape(layer2.shape[1], layer2.shape[2])

# Apply thresholds
if normalize:
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())

anomaly_map = (anomaly_map > pixel_threshold).float() * 255
anomaly_map = anomaly_map.cpu().numpy().astype(np.uint8)

# ================= SAVE OUTPUT =================
input_filename = os.path.basename(IMAGE_PATH)
output_filename = os.path.splitext(input_filename)[0] + "_anomaly_map.png"
output_path = os.path.join(SAVE_DIR, output_filename)

cv2.imwrite(output_path, anomaly_map)
print(f"Anomaly map saved to: {output_path}")