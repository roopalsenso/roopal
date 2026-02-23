# accuracydrop_segmentation_with_fg_bg_percentage.py
import os
import glob
import cv2
import numpy as np
import onnxruntime as ort

# ---------------- CONFIG ----------------
models = [
    "/root/old-data/home/roopal/models/fp32/model_fp32.onnx",
    "/root/old-data/home/roopal/models/fp16/model_fp16.onnx",
    "/root/old-data/home/roopal/models/int8/model_int8.onnx"
]

dataset_images = "/root/old-data/home/roopal/datasets/test_offline_dataset/images/*.jpg"
dataset_labels = "/root/old-data/home/roopal/datasets/test_offline_dataset/annotations/*.png"

device_type = "CPU"
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

# ---------------- CALCULATE FOREGROUND PERCENTAGE ----------------
total_pixels = 0
foreground_pixels = 0
for lbl_path in label_files:
    mask = cv2.imread(lbl_path, 0)
    total_pixels += mask.size
    foreground_pixels += np.sum(mask == 1)

foreground_percentage = (foreground_pixels / total_pixels) * 100
background_percentage = 100 - foreground_percentage
print(f"Foreground pixel percentage in dataset: {foreground_percentage:.4f}%")
print(f"Background pixel percentage in dataset: {background_percentage:.4f}%\n")

# ---------------- RUN MODELS ----------------
# First compute FP32 baseline for accuracy drop
fp32_session = ort.InferenceSession(models[0], providers=["CPUExecutionProvider"])
input_name = fp32_session.get_inputs()[0].name
H, W = fp32_session.get_inputs()[0].shape[2:4]

total_iou_fp32 = 0
total_bg_iou_fp32 = 0
total_fg_iou_fp32 = 0

for img_path, lbl_path in zip(image_files, label_files):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H)).astype(np.float32) / 255.0
    img = np.transpose(img, (2,0,1))[None,...]

    pred = fp32_session.run(None, {input_name: img.astype(np.float32)})[0]
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

print(f"FP32 Results:")
print(f"Average Background IoU: {baseline_bg_iou:.4f}")
print(f"Average Foreground IoU: {baseline_fg_iou:.4f}")
print(f"FP32 Mean IoU: {baseline_iou:.4f}\n")

# ---------------- RUN QUANTIZED MODELS ----------------
for model_path in models[1:]:
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_type = session.get_inputs()[0].type

    total_iou = 0
    total_bg_iou = 0
    total_fg_iou = 0

    for img_path, lbl_path in zip(image_files, label_files):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W,H)).astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))[None,...]

        if input_type == 'tensor(int8)':
            img_input = (img * 255).astype(np.int8)
        elif input_type == 'tensor(float16)':
            img_input = img.astype(np.float16)
        else:
            img_input = img.astype(np.float32)

        output = session.run(None, {input_name: img_input})[0]
        pred = output[0]
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
    accuracy_drop = (baseline_iou - avg_iou) / baseline_iou * 100

    quant_type = "INT8" if "int8" in model_path.lower() else "FP16"
    # ---------------- PRINT RESULTS ----------------
    print(f"\n{quant_type} Model Results:")
    print(f"Average Background IoU: {avg_bg_iou:.4f}")
    print(f"Average Foreground IoU: {avg_fg_iou:.4f}")
    # ---------------- KEEP ORIGINAL LINES ----------------
    print(f"{quant_type} Model Average IoU: {avg_iou:.4f}")
    print(f"Accuracy drop vs FP32: {accuracy_drop:.2f}%\n")
