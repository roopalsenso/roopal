import pandas as pd
from tabulate import tabulate

# =========================
# CSV PATHS
# =========================
csv1 = "/root/old-data/home/roopal/inference/results/anomaly/onnx_scores.csv"
csv2 = "/root/old-data/home/roopal/inference/results/anomaly/trt_fp16_scores.csv"
csv3 = "/root/old-data/home/roopal/inference/results/anomaly/trt_fp32_scores.csv"
csv4 = "/root/old-data/home/roopal/inference/results/anomaly/trtint8_scores.csv"
csv5 = "/root/old-data/home/roopal/inference/results/anomaly/padim_prediction_results.csv"

save_path = "/root/old-data/home/roopal/inference/results/anomaly/comparison.csv"

# =========================
# LOAD DATA
# =========================
onnx = pd.read_csv(csv1)
fp16 = pd.read_csv(csv2)
fp32 = pd.read_csv(csv3)
int8 = pd.read_csv(csv4)
padim = pd.read_csv(csv5)

# =========================
# KEEP REQUIRED COLUMNS
# =========================
onnx = onnx[["Image Name","Anomaly Score"]].rename(columns={"Anomaly Score":"ONNX"})
fp16 = fp16[["Image Name","Anomaly Score"]].rename(columns={"Anomaly Score":"FP16"})
fp32 = fp32[["Image Name","Anomaly Score"]].rename(columns={"Anomaly Score":"FP32"})
int8 = int8[["Image Name","Anomaly Score"]].rename(columns={"Anomaly Score":"INT8"})
padim = padim[["Image Name","Anomaly Score"]].rename(columns={"Anomaly Score":"PADIM"})

# =========================
# MERGE
# =========================
df = onnx.merge(fp16,on="Image Name")
df = df.merge(fp32,on="Image Name")
df = df.merge(int8,on="Image Name")
df = df.merge(padim,on="Image Name")

# =========================
# DIFFERENCES FROM ONNX
# =========================
df["Diff_FP16"] = abs(df["ONNX"] - df["FP16"])
df["Diff_FP32"] = abs(df["ONNX"] - df["FP32"])
df["Diff_INT8"] = abs(df["ONNX"] - df["INT8"])
df["Diff_PADIM"] = abs(df["ONNX"] - df["PADIM"])

# =========================
# SAVE CSV
# =========================
df.to_csv(save_path,index=False)

print("\nComparison Table:\n")
print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))

print("\nSaved comparison CSV at:", save_path)