import os
import glob
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from PIL import Image
import torchvision.transforms as T

# ==============================
# CONFIG
# ==============================
ONNX_PATH = "/root/old-data/home/roopal/models/onnx/patchcore_backbone_op13_static.onnx"
IMAGE_DIR = "/root/old-data/home/roopal/datasets/anomaly_calib"   # <-- change if needed
CACHE_FILE = "/root/old-data/home/roopal/datasets/anomaly_calibration.cache"
BATCH_SIZE = 1

# ==============================
# PATCHCORE EXACT PREPROCESSING
# ==============================
transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

print("============================================")
print("Calibration Script Started")
print("ONNX Model Path:", os.path.abspath(ONNX_PATH))
print("Calibration Image Folder:", os.path.abspath(IMAGE_DIR))
print("Calibration Cache Will Be Saved At:", os.path.abspath(CACHE_FILE))
print("============================================")

# ==============================
# CALIBRATOR CLASS
# ==============================
class PatchCoreCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, image_dir, batch_size=1, cache_file="calibration.cache"):
        super().__init__()
        self.batch_size = batch_size
        self.cache_file = cache_file

        self.image_paths = glob.glob(os.path.join(image_dir, "*"))
        self.current_index = 0

        print(f"[INFO] Total images found: {len(self.image_paths)}")
        if len(self.image_paths) == 0:
            raise RuntimeError("No images found for calibration.")

        self.device_input = cuda.mem_alloc(
            batch_size * 3 * 224 * 224 * np.float32().nbytes
        )

        print("[INFO] Device memory allocated for calibration batch.")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.image_paths):
            print("[INFO] All calibration images processed.")
            return None

        batch = []

        for _ in range(self.batch_size):
            img_path = self.image_paths[self.current_index]
            print(f"[INFO] Processing image: {img_path}")
            self.current_index += 1

            img = Image.open(img_path).convert("RGB")
            img = transform(img)
            img = img.numpy()

            print("      -> Preprocessing Applied:")
            print("         Resize(256,256)")
            print("         CenterCrop(224,224)")
            print("         RGB conversion")
            print("         Normalize(mean,std)")
            print("         Output shape:", img.shape)

            batch.append(img)

        batch = np.ascontiguousarray(batch).astype(np.float32)
        cuda.memcpy_htod(self.device_input, batch)

        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print("[INFO] Existing calibration cache found. Using it.")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        print("[INFO] Writing calibration cache to:", os.path.abspath(self.cache_file))
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print("[INFO] Calibration cache saved successfully.")


# ==============================
# BUILD ENGINE FOR CALIBRATION
# ==============================
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

print("\n[INFO] Creating TensorRT Builder...")
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)
parser = trt.OnnxParser(network, TRT_LOGGER)

print("[INFO] Parsing ONNX model...")
with open(ONNX_PATH, "rb") as model:
    if not parser.parse(model.read()):
        print("[ERROR] Failed to parse ONNX.")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        raise RuntimeError("ONNX parsing failed.")
print("[INFO] ONNX model parsed successfully.")

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

config.set_flag(trt.BuilderFlag.INT8)

calibrator = PatchCoreCalibrator(
    image_dir=IMAGE_DIR,
    batch_size=BATCH_SIZE,
    cache_file=CACHE_FILE
)

config.int8_calibrator = calibrator

print("\n[INFO] Starting INT8 calibration (this may take a few minutes)...")

serialized_engine = builder.build_serialized_network(network, config)

if serialized_engine is None:
    raise RuntimeError("Engine build failed during calibration.")

print("[INFO] Engine built successfully during calibration phase.")
print("\n============================================")
print("Calibration Complete.")
print("Calibration cache stored at:", os.path.abspath(CACHE_FILE))
print("You can now use trtexec with this cache.")
print("============================================")