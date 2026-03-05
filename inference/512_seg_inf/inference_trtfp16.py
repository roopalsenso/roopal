import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os

# ================= CONFIG =================
ENGINE_PATH = "/root/old-data/home/roopal/engines/semsons512fp16.trt"
IMAGE_PATH  = "/root/old-data/home/roopal/datasets/test_offline_dataset/images/13.jpg"
SAVE_DIR    = "/root/old-data/home/roopal/inference/results/fp16trt"

INPUT_H = 512
INPUT_W = 512
WARMUP_RUNS = 10
MEASURE_RUNS = 50
# =========================================

os.makedirs(SAVE_DIR, exist_ok=True)

# -------- TensorRT setup --------
logger = trt.Logger(trt.Logger.INFO)
runtime = trt.Runtime(logger)

with open(ENGINE_PATH, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

input_name  = engine.get_tensor_name(0)
output_name = engine.get_tensor_name(1)

input_shape  = tuple(engine.get_tensor_shape(input_name))
output_shape = tuple(engine.get_tensor_shape(output_name))

print(f"Input  : {input_name}  {input_shape}  {engine.get_tensor_dtype(input_name)}")
print(f"Output : {output_name} {output_shape} {engine.get_tensor_dtype(output_name)}")

# -------- Image preprocessing --------
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (INPUT_W, INPUT_H))

img_input = img.astype(np.float32) / 255.0
img_input = np.transpose(img_input, (2, 0, 1))  # HWC -> CHW
img_input = np.expand_dims(img_input, axis=0)   # NCHW
img_input = np.ascontiguousarray(img_input)

# -------- Memory allocation --------
d_input = cuda.mem_alloc(img_input.nbytes)

dtype_map = {
    trt.DataType.FLOAT: np.float32,
    trt.DataType.HALF:  np.float16,
    trt.DataType.INT32: np.int32,
    trt.DataType.INT8:  np.int8
}

output_dtype = dtype_map[engine.get_tensor_dtype(output_name)]
output_size  = int(np.prod(output_shape) * np.dtype(output_dtype).itemsize)
d_output = cuda.mem_alloc(output_size)

output = np.empty(output_shape, dtype=output_dtype)

bindings = [int(d_input), int(d_output)]

# -------- Warm-up --------
print("Warming up...")
for _ in range(WARMUP_RUNS):
    cuda.memcpy_htod(d_input, img_input)
    context.execute_v2(bindings)
    cuda.memcpy_dtoh(output, d_output)

# -------- Timing --------
gpu_times = []
full_times = []

print("Measuring inference time...")

for _ in range(MEASURE_RUNS):

    # ---- GPU COMPUTE ONLY ----
    start_gpu = cuda.Event()
    end_gpu   = cuda.Event()

    start_gpu.record()
    context.execute_v2(bindings)
    end_gpu.record()
    end_gpu.synchronize()

    gpu_times.append(start_gpu.time_till(end_gpu))

    # ---- FULL PIPELINE (H2D + GPU + D2H) ----
    start_full = cuda.Event()
    end_full   = cuda.Event()

    start_full.record()
    cuda.memcpy_htod(d_input, img_input)
    context.execute_v2(bindings)
    cuda.memcpy_dtoh(output, d_output)
    end_full.record()
    end_full.synchronize()

    full_times.append(start_full.time_till(end_full))

# -------- Results --------
gpu_times  = np.array(gpu_times)
full_times = np.array(full_times)

print("\n===== PERFORMANCE SUMMARY =====")
print(f"GPU Compute Time  : {gpu_times.mean():.2f} ms")
print(f"Full Pipeline Time (H2D + Compute + D2H): {full_times.mean():.2f} ms")
print(f"FPS (compute only): {1000.0 / gpu_times.mean():.2f}")
print("================================\n")

# -------- Post-processing --------
# Remove batch dimension
output = output[0]  # (2, 512, 512)

# Convert to float32 if needed
if output.dtype == np.float16:
    output = output.astype(np.float32)

# Take argmax over class dimension
mask = np.argmax(output, axis=0)  # (512, 512)

# Convert to uint8
mask = (mask * 255).astype(np.uint8)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
overlay = img.copy()
cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)

# Get base filename without extension
base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

mask_path = os.path.join(SAVE_DIR, f"{base_name}_mask.jpg")
overlay_path = os.path.join(SAVE_DIR, f"{base_name}_overlay.jpg")

cv2.imwrite(mask_path, mask)
cv2.imwrite(overlay_path, overlay)

print(f"Saved mask to: {mask_path}")
print(f"Saved overlay to: {overlay_path}")

print("Results saved to:", SAVE_DIR)