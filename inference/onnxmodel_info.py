import onnx
import os
from onnx import numpy_helper

ONNX_PATH = "/root/old-data/home/roopal/models/onnx/patchcore_backbone_op13_static.onnx"  # change if needed

def get_shape(tensor):
    shape = []
    for dim in tensor.type.tensor_type.shape.dim:
        if dim.dim_value > 0:
            shape.append(dim.dim_value)
        else:
            shape.append("dynamic")
    return shape

def get_dtype(tensor):
    return tensor.type.tensor_type.elem_type

def main():
    print("="*60)
    print("Loading ONNX model...")
    model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(model)
    print("Model loaded successfully.")
    print("="*60)

    # Basic Info
    print("\n📌 BASIC MODEL INFO")
    print("IR Version:", model.ir_version)
    print("Producer Name:", model.producer_name)
    print("Producer Version:", model.producer_version)

    print("\n📌 OPSET VERSION")
    for opset in model.opset_import:
        print("Domain:", opset.domain if opset.domain else "default",
              "| Version:", opset.version)

    # Inputs
    print("\n📌 INPUTS")
    for i, input_tensor in enumerate(model.graph.input):
        print(f"\nInput {i+1}")
        print("Name:", input_tensor.name)
        print("Shape:", get_shape(input_tensor))
        print("Data Type:", get_dtype(input_tensor))

    # Outputs
    print("\n📌 OUTPUTS")
    for i, output_tensor in enumerate(model.graph.output):
        print(f"\nOutput {i+1}")
        print("Name:", output_tensor.name)
        print("Shape:", get_shape(output_tensor))
        print("Data Type:", get_dtype(output_tensor))

    # Initializers (Weights)
    print("\n📌 INITIALIZERS (Weights)")
    total_params = 0
    for init in model.graph.initializer:
        array = numpy_helper.to_array(init)
        total_params += array.size
    print("Total parameter tensors:", len(model.graph.initializer))
    print("Total parameters:", total_params)

    # Nodes
    print("\n📌 GRAPH NODES")
    print("Total nodes:", len(model.graph.node))

    # Model size
    size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    print("\n📌 MODEL FILE SIZE: {:.2f} MB".format(size_mb))

    print("\nInspection Complete.")
    print("="*60)

if __name__ == "__main__":
    main()