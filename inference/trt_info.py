import tensorrt as trt
import os

ENGINE_PATH = "/root/old-data/home/roopal/engines/anomaly/anomalyint8_backbone.trt"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def dtype_to_string(dtype):
    return str(dtype).replace("DataType.", "")


def print_profiles(engine):
    print("\n----- OPTIMIZATION PROFILES -----")
    for i in range(engine.num_optimization_profiles):
        print(f"\nProfile {i}:")
        for t in range(engine.num_io_tensors):
            name = engine.get_tensor_name(t)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                min_shape, opt_shape, max_shape = engine.get_tensor_profile_shape(name, i)
                print(f"  Input: {name}")
                print(f"    MIN: {min_shape}")
                print(f"    OPT: {opt_shape}")
                print(f"    MAX: {max_shape}")
    print("---------------------------------\n")


def print_engine_info(engine):
    print("\n========== ENGINE INFORMATION ==========")
    print(f"Engine Name            : {engine.name}")
    print(f"TensorRT Version       : {trt.__version__}")
    print(f"Number of IO Tensors   : {engine.num_io_tensors}")
    print(f"Number of Profiles     : {engine.num_optimization_profiles}")
    print(f"Engine Capability      : {engine.engine_capability}")
    print(f"Refittable             : {engine.refittable}")
    print("========================================\n")

    print("---------- TENSOR DETAILS ----------")

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        location = engine.get_tensor_location(name)

        print(f"\nTensor Name : {name}")
        print(f"Mode        : {'INPUT' if mode == trt.TensorIOMode.INPUT else 'OUTPUT'}")
        print(f"Data Type   : {dtype_to_string(dtype)}")
        print(f"Shape       : {list(shape)}")
        print(f"Location    : {location}")

        dynamic = any(dim == -1 for dim in shape)
        print(f"Dynamic     : {'YES' if dynamic else 'NO'}")

    print("\n=====================================")

    if engine.num_optimization_profiles > 0:
        print_profiles(engine)


def main():
    if not os.path.exists(ENGINE_PATH):
        print("Engine file not found.")
        return

    with open(ENGINE_PATH, "rb") as f:
        engine_data = f.read()

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_data)

    if engine is None:
        print("Failed to deserialize engine.")
        return

    print_engine_info(engine)


if __name__ == "__main__":
    main()