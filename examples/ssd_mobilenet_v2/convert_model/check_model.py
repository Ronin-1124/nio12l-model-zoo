import onnx

model = onnx.load("./ssd_mobilenet_v2.onnx")

print("Inputs:")
for inp in model.graph.input:
    t = inp.type.tensor_type
    shape = []
    for d in t.shape.dim:
        if d.dim_value:
            shape.append(d.dim_value)
        else:
            shape.append(d.dim_param)
    print(" ", inp.name, "dtype:", t.elem_type, "shape:", shape)

print("\nOutputs:")
for out in model.graph.output:
    t = out.type.tensor_type
    shape = []
    for d in t.shape.dim:
        if d.dim_value:
            shape.append(d.dim_value)
        else:
            shape.append(d.dim_param)
    print(" ", out.name, "dtype:", t.elem_type, "shape:", shape)
