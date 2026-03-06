import os
from pathlib import Path
import csv
import onnxruntime as ort
import numpy as np
import cv2
import json

# -------------------------------
# Paths and configs
# -------------------------------
ONNX_PATH = "/root/old-data/home/roopal/models/onnx/patchcore_backbone_op13_static.onnx"
MEMORY_BANK_PATH = "/root/old-data/home/roopal/models/onnx/patchcore_memory_bank.npy"
METADATA_PATH = "/root/old-data/home/roopal/models/onnx/patchcore_metadata.json"
DATASET_DIR = "/root/old-data/home/roopal/datasets/MvTecAD/MvTecAD/OnlyOkbottles_test/images"
RESULT_DIR = "/root/old-data/home/roopal/inference/results/anomaly"
NUM_IMAGES = 20
INPUT_H, INPUT_W = 224, 224
K = 9

os.makedirs(RESULT_DIR, exist_ok=True)

# -------------------------------
# Load memory bank
memory_bank = np.load(MEMORY_BANK_PATH).astype(np.float32)
memory_bank = memory_bank / (np.linalg.norm(memory_bank, axis=1, keepdims=True) + 1e-8)

# Load metadata
with open(METADATA_PATH) as f:
    meta = json.load(f)

# ONNX session
ort_session = ort.InferenceSession(ONNX_PATH)

# -------------------------------
def run_onnx_inference(img_path: str) -> float:
    """Run ONNX PatchCore backbone + Python postprocessing and return anomaly score."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256,256))
    start = (256-224)//2
    img = img[start:start+224, start:start+224]
    img = img.astype(np.float32)/255.0

    mean = np.array([0.485,0.456,0.406])
    std  = np.array([0.229,0.224,0.225])
    img = (img - mean)/std
    img = np.transpose(img,(2,0,1))
    img = np.expand_dims(img,0).astype(np.float32)

    # ONNX inference
    input_name = ort_session.get_inputs()[0].name
    feat2, feat3 = ort_session.run(None, {input_name: img})

    # Upsample layer3
    upsampled = []
    for c in range(feat3.shape[1]):
        upsampled.append(cv2.resize(feat3[0,c], (feat2.shape[2], feat2.shape[3]), interpolation=cv2.INTER_LINEAR))
    feat3 = np.stack(upsampled, axis=0)

    # Concatenate + reshape
    embedding = np.concatenate([feat2[0], feat3], axis=0).reshape(feat2.shape[1]+feat3.shape[0], -1).T
    embedding = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8)

    # KNN distance
    emb_norm = np.sum(embedding**2, axis=1, keepdims=True)
    mem_norm = np.sum(memory_bank**2, axis=1)
    dists = emb_norm + mem_norm - 2*np.dot(embedding, memory_bank.T)
    dists = np.maximum(dists,0)
    dists = np.sqrt(dists)
    knn = np.partition(dists,K,axis=1)[:,:K]
    patch_scores = np.mean(knn,axis=1)
    return patch_scores.max()

# -------------------------------
# Collect image paths
image_paths = []
for defect in ["good", "contamination", "broken_small", "broken_large"]:
    folder = Path(DATASET_DIR) / defect
    image_paths.extend(list(folder.glob("*.png")))
image_paths = image_paths[:NUM_IMAGES]

# -------------------------------
# -------------------------------
# Iterate over test folders and run inference
results = []

THRESHOLD = 0.5  # adjust if needed

test_root = Path(DATASET_DIR)

for class_dir in test_root.iterdir():

    if class_dir.is_dir():

        for image_path in class_dir.glob("*.png"):

            score = run_onnx_inference(str(image_path))
            pred_label = score > THRESHOLD

            print(f"[{class_dir.name}/{image_path.name}] → Anomaly Score: {score:.4f}, Pred Label: {pred_label}")

            results.append([
                class_dir.name,
                image_path.name,
                f"{score:.4f}",
                pred_label
            ])

# -------------------------------
# Save CSV
csv_path = os.path.join(RESULT_DIR, "onnx_scores.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image Folder", "Image Name", "Anomaly Score", "Predicted Label"])
    writer.writerows(results)

print(f"\nSaved results to {csv_path}")