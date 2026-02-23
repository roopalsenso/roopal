# accuracydrop_fp32_vs_trtint8.py
#old-data/home/roopal/scripts/onnx_quant/accuracydrop.py
import os
import glob
import cv2
import numpy as np
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

# ---------------- CONFIG ----------------
models = [
    "/root/old-data/home/roopal/models/fp32/512_2_semsons_mode_512_color_simplified.onnx",
    "/root/old-data/home/roopal/engines/semsons512int8.trt"
]

dataset_images = "/root/old-data/home/roopal/datasets/test_offline_dataset/images/*.jpg"
dataset_labels = "/root/old-data/home/roopal/datasets/test_offline_dataset/annotations/*.png"

num_classes = 2  # class 0 = background, class 1 = foreground

# ---------------- HELPER: IoU ----------------
def compute_class_iou(pred, target, cls):
    pred_inds = (pred == cls)
    target_inds = (target == cls)
    intersection = np.logical_and(pred_inds, target_inds).sum()
    union = np.logical_or(pred_inds, target_inds).sum()
    return intersection / union if union != 0 else 1.0

def compute_mean_iou(pred, target, num_classes):
    return np.mean([compute_class_iou(pred, target, cls) for cls in range(num_classes)])

# ---------------- GET DATA FILES ----------------
image_files = sorted(glob.glob(dataset_images))
label_files = sorted(glob.glob(dataset_labels))
assert len(image_files) == len(label_files), "Mismatch between images and labels"

print(f"Total images found: {len(image_files)}\n")

# ---------------- RUN FP32 ONNX ----------------
fp32_session = ort.InferenceSession(models[0], providers=["CPUExecutionProvider"])
input_name = fp32_session.get_inputs()[0].name
H, W = fp32_session.get_inputs()[0].shape[2:4]

total_iou_fp32 = total_bg_iou_fp32 = total_fg_iou_fp32 = 0

for img_path, lbl_path in zip(image_files, label_files):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H)).astype(np.float32) / 255.0
    img_input = np.transpose(img, (2,0,1))[None,...]

    pred = fp32_session.run(None, {input_name: img_input.astype(np.float32)})[0]
    pred = pred[0]
    if len(pred.shape) == 3 and pred.shape[0] == num_classes:
        pred = np.argmax(pred, axis=0)

    target = cv2.imread(lbl_path, 0)
    target_resized = cv2.resize(target, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)

    total_iou_fp32 += compute_mean_iou(pred, target_resized, num_classes)
    total_bg_iou_fp32 += compute_class_iou(pred, target_resized, 0)
    total_fg_iou_fp32 += compute_class_iou(pred, target_resized, 1)

baseline_iou = total_iou_fp32 / len(image_files)
baseline_bg_iou = total_bg_iou_fp32 / len(image_files)
baseline_fg_iou = total_fg_iou_fp32 / len(image_files)

print(f"FP32 ONNX Results (vs Ground Truth):")
print(f"Average Background IoU: {baseline_bg_iou:.4f}")
print(f"Average Foreground IoU: {baseline_fg_iou:.4f}")
print(f"FP32 Mean IoU: {baseline_iou:.4f}\n")

# ---------------- RUN INT8 TRT ----------------
trt_logger = trt.Logger(trt.Logger.INFO)
with open(models[1], "rb") as f:
    engine_data = f.read()

runtime = trt.Runtime(trt_logger)
engine = runtime.deserialize_cuda_engine(engine_data)
context = engine.create_execution_context()

input_name_trt = engine.get_tensor_name(0)
output_name_trt = engine.get_tensor_name(1)
input_shape_trt = tuple(engine.get_tensor_shape(input_name_trt))
output_shape_trt = tuple(engine.get_tensor_shape(output_name_trt))
output_dtype_trt = np.int32  # usually INT8 engine output is int32

total_iou_trt = total_bg_iou_trt = total_fg_iou_trt = 0

for img_path, lbl_path in zip(image_files, label_files):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (input_shape_trt[3], input_shape_trt[2])).astype(np.float32)/255.0
    img_input = np.transpose(img_resized, (2,0,1))[None,...].astype(np.float32)
    img_input = np.ascontiguousarray(img_input)

    # Allocate GPU memory
    d_input = cuda.mem_alloc(int(img_input.nbytes))
    d_output = cuda.mem_alloc(int(np.prod(output_shape_trt) * np.dtype(output_dtype_trt).itemsize))
    cuda.memcpy_htod(d_input, img_input)

    # Run inference
    context.execute_v2([int(d_input), int(d_output)])

    # Copy output back
    output_trt = np.empty(output_shape_trt, dtype=output_dtype_trt)
    cuda.memcpy_dtoh(output_trt, d_output)

    # Convert to 0/1 mask
    pred_trt = output_trt[0]
    if len(pred_trt.shape) == 3 and pred_trt.shape[0] == num_classes:
        pred_trt = np.argmax(pred_trt, axis=0)

    target = cv2.imread(lbl_path, 0)
    target_resized = cv2.resize(target, (pred_trt.shape[1], pred_trt.shape[0]), interpolation=cv2.INTER_NEAREST)

    total_iou_trt += compute_mean_iou(pred_trt, target_resized, num_classes)
    total_bg_iou_trt += compute_class_iou(pred_trt, target_resized, 0)
    total_fg_iou_trt += compute_class_iou(pred_trt, target_resized, 1)

avg_iou_trt = total_iou_trt / len(image_files)
avg_bg_iou_trt = total_bg_iou_trt / len(image_files)
avg_fg_iou_trt = total_fg_iou_trt / len(image_files)
accuracy_drop_trt = (baseline_iou - avg_iou_trt) / baseline_iou * 100

print(f"\nINT8 TRT Results (vs Ground Truth):")
print(f"Average Background IoU: {avg_bg_iou_trt:.4f}")
print(f"Average Foreground IoU: {avg_fg_iou_trt:.4f}")
print(f"INT8 TRT Mean IoU: {avg_iou_trt:.4f}")
print(f"Accuracy drop vs FP32: {accuracy_drop_trt:.2f}%")