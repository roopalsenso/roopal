import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# -------- CONFIG --------
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
INPUT_H, INPUT_W = 512, 512
DATASET_PATH = "/root/old-data/home/roopal/datasets/512_2_clstalbros_tets_color"

# -------- HELPERS --------
def load_engine(engine_path):
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(f.read())

def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_W, INPUT_H))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return np.ascontiguousarray(img)

def get_predictions(engine_path):
    """Return predictions for all images as a dict: img_name -> (pred_class, confidence)"""
    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    input_size = trt.volume([1, 3, 512, 512])
    output_size = trt.volume([1, 2])

    d_input = cuda.mem_alloc(input_size * np.float32().nbytes)
    d_output = cuda.mem_alloc(output_size * np.float32().nbytes)
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    stream = cuda.Stream()
    output_host = np.empty(output_size, dtype=np.float32)

    predictions = {}

    for img_name in os.listdir(DATASET_PATH):
        img_path = os.path.join(DATASET_PATH, img_name)
        input_data = preprocess(img_path)

        cuda.memcpy_htod_async(d_input, input_data, stream)
        context.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(output_host, d_output, stream)
        stream.synchronize()

        pred_class = int(np.argmax(output_host))
        confidence = float(output_host[pred_class])
        predictions[img_name] = (pred_class, confidence)

    return predictions

def compare_accuracy(baseline_preds, compare_preds):
    """Compute accuracy of compare_preds relative to baseline_preds"""
    correct = 0
    total = 0
    for img_name, (baseline_cls, _) in baseline_preds.items():
        compare_cls, _ = compare_preds.get(img_name, (-1, 0))
        if compare_cls == baseline_cls:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0

# -------- GET PREDICTIONS --------
fp32_preds = get_predictions("/root/old-data/home/roopal/engines/512_2_cls_talbros/clsfp32.trt")
fp16_preds = get_predictions("/root/old-data/home/roopal/engines/512_2_cls_talbros/clsfp16.trt")
int8_preds = get_predictions("/root/old-data/home/roopal/engines/512_2_cls_talbros/clsint8.trt")

# -------- PRINT PER-IMAGE COMPARISON --------
print("\n===== PER IMAGE PREDICTIONS =====")
print(f"{'Image':40s} | {'FP32 (class,conf)':20s} | {'FP16 (class,conf)':20s} | {'INT8 (class,conf)':20s}")
print("-"*110)
for img_name in fp32_preds.keys():
    fp32_cls, fp32_conf = fp32_preds[img_name]
    fp16_cls, fp16_conf = fp16_preds[img_name]
    int8_cls, int8_conf = int8_preds[img_name]

    print(f"{img_name:40s} | ({fp32_cls},{fp32_conf:.3f}){'':8s} | ({fp16_cls},{fp16_conf:.3f}){'':8s} | ({int8_cls},{int8_conf:.3f})")

# -------- ACCURACY COMPARISON w.r.t FP32 BASELINE --------
fp16_acc = compare_accuracy(fp32_preds, fp16_preds)
int8_acc = compare_accuracy(fp32_preds, int8_preds)

print("\n===== ACCURACY COMPARISON w.r.t FP32 BASELINE =====")
print(f"FP16 matches FP32: {fp16_acc*100:.2f}%")
print(f"INT8 matches FP32: {int8_acc*100:.2f}%")