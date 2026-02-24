#roopal/inference/inference_trtfp16.py
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
import time

ENGINE_PATH = "/root/old-data/home/roopal/engines/ppliteseg_fp16.trt"
IMAGE_PATH = "/root/old-data/home/roopal/datasets/test_offline_dataset/images/2.jpg"
SAVE_DIR = "/root/old-data/home/roopal/inference/results/fp16trt"
os.makedirs(SAVE_DIR, exist_ok=True)

logger = trt.Logger(trt.Logger.INFO)
runtime = trt.Runtime(logger)

with open(ENGINE_PATH, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

print(f"Input tensor: {engine.get_tensor_name(0)} {tuple(engine.get_tensor_shape(engine.get_tensor_name(0)))} "
      f"DataType.{engine.get_tensor_dtype(engine.get_tensor_name(0)).name}")
print(f"Output tensor: {engine.get_tensor_name(1)} {tuple(engine.get_tensor_shape(engine.get_tensor_name(1)))} "
      f"DataType.{engine.get_tensor_dtype(engine.get_tensor_name(1)).name}")

img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (512, 512))
img_input = img.astype(np.float32) / 255.0
img_input = np.transpose(img_input, (2, 0, 1))
img_input = np.expand_dims(img_input, axis=0)
img_input = img_input.copy()

input_shape = tuple(engine.get_tensor_shape(engine.get_tensor_name(0)))
output_shape = tuple(engine.get_tensor_shape(engine.get_tensor_name(1)))

d_input = cuda.mem_alloc(img_input.nbytes)

dtype_map = {trt.DataType.FLOAT: np.float32,
             trt.DataType.HALF: np.float16,
             trt.DataType.INT32: np.int32,
             trt.DataType.INT8: np.int8}

output_dtype = dtype_map[engine.get_tensor_dtype(engine.get_tensor_name(1))]
d_output = cuda.mem_alloc(int(np.prod(output_shape) * np.dtype(output_dtype).itemsize))

cuda.memcpy_htod(d_input, img_input)

start = time.time()
context.execute_v2([int(d_input), int(d_output)])
end = time.time()
inference_time = (end - start) * 1000

output = np.empty(output_shape, dtype=output_dtype)
cuda.memcpy_dtoh(output, d_output)

mask_valid = output[0]
if mask_valid.dtype == np.float16:
    mask_valid = mask_valid.astype(np.float32)
mask_valid = (mask_valid * 255).astype(np.uint8)

contours, _ = cv2.findContours(mask_valid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
overlay = img.copy()
cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

cv2.imwrite(f"{SAVE_DIR}/overlay_result.jpg", overlay)
cv2.imwrite(f"{SAVE_DIR}/mask_result.jpg", mask_valid)

print("Saved results to:", SAVE_DIR)
print(f"Inference time: {inference_time:.2f} ms")
