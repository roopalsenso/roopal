#roopal/inference/inference_trtint8.py
import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

ENGINE_PATH = "/root/old-data/home/roopal/engines/ppliteseg_int8_nocalib.trt"
IMAGE_PATH = "/root/old-data/home/roopal/datasets/test_offline_dataset/images/2.jpg"
SAVE_DIR = "/root/old-data/home/roopal/inference/results/int8trt_nocalib"
os.makedirs(SAVE_DIR, exist_ok=True)

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

with open(ENGINE_PATH, "rb") as f:
    serialized_engine = f.read()

runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(serialized_engine)
context = engine.create_execution_context()

input_name = engine.get_tensor_name(0)
output_name = engine.get_tensor_name(1)

input_shape = tuple(engine.get_tensor_shape(input_name))
output_shape = tuple(engine.get_tensor_shape(output_name))

print(f"[INFO] Input tensor: {input_name}, shape={input_shape}")
print(f"[INFO] Output tensor: {output_name}, shape={output_shape}, dtype={engine.get_tensor_dtype(output_name)}")

# -------------------------------
# Prepare input
# -------------------------------
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (input_shape[3], input_shape[2]))
img_input = img.astype(np.float32) / 255.0
img_input = np.transpose(img_input, (2, 0, 1))
img_input = np.expand_dims(img_input, axis=0)
img_input = np.ascontiguousarray(img_input)

# Allocate GPU memory
d_input = cuda.mem_alloc(img_input.nbytes)
output_dtype = np.int32  # INT32 output from trtexec INT8 engine
d_output = cuda.mem_alloc(int(np.prod(output_shape) * np.dtype(output_dtype).itemsize))

cuda.memcpy_htod(d_input, img_input)

# -------------------------------
# Run inference
# -------------------------------
start = time.time()
context.execute_v2([int(d_input), int(d_output)])
end = time.time()
inference_time = (end - start) * 1000

# Copy output back
output = np.empty(output_shape, dtype=output_dtype)
cuda.memcpy_dtoh(output, d_output)

# -------------------------------
# Post-process mask
# -------------------------------
mask = output[0].astype(np.uint8)  # batch=1
mask_uint8 = mask * 255

# Draw contours
overlay = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

# Save results
cv2.imwrite(f"{SAVE_DIR}/mask.jpg", mask_uint8)
cv2.imwrite(f"{SAVE_DIR}/overlay.jpg", overlay)

print(f"[INFO] Saved results to {SAVE_DIR}")
print(f"[INFO] Inference time: {inference_time:.2f} ms")
