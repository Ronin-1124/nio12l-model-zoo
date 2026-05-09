import copy
import onnx
from onnx import helper, TensorProto

src_model = "./ssd_mobilenet_v2.onnx"
dst_model = "./ssd_mobilenet_v2_core.onnx"

first_conv_name = (
    "StatefulPartitionedCall/"
    "ssd_mobile_net_v2keras_feature_extractor/"
    "functional_1/Conv1/Conv2D"
)

new_input_name = "preprocessed_input"

output_names = [
    "raw_detection_boxes",
    "raw_detection_scores",
]

model = onnx.load(src_model)
graph = model.graph

# 1. Locate the first Conv and replace its feature input.
first_conv = None
for node in graph.node:
    if node.name == first_conv_name:
        first_conv = node
        break

if first_conv is None:
    raise RuntimeError(f"Cannot find first Conv node: {first_conv_name}")

old_conv_input = first_conv.input[0]
print("First Conv:", first_conv.name)
print("Old Conv input:", old_conv_input)

first_conv.input[0] = new_input_name

# 2. Traverse backward from raw outputs and keep required nodes.
nodes = list(graph.node)

required_tensors = set(output_names)
required_node_indices = set()

for idx in range(len(nodes) - 1, -1, -1):
    node = nodes[idx]

    if any(out in required_tensors for out in node.output):
        required_node_indices.add(idx)

        for inp in node.input:
            if inp:
                required_tensors.add(inp)

kept_nodes = [
    copy.deepcopy(nodes[i]) for i in range(len(nodes)) if i in required_node_indices
]

print("Original node count:", len(nodes))
print("Kept node count:", len(kept_nodes))

# 3. Check whether unresolved external dependencies remain.
initializer_names = {init.name for init in graph.initializer}
produced_names = set()
for node in kept_nodes:
    for out in node.output:
        produced_names.add(out)

allowed_external_inputs = {new_input_name}

unresolved = (
    required_tensors - initializer_names - produced_names - allowed_external_inputs
)

# Drop empty strings.
unresolved = {x for x in unresolved if x}

if unresolved:
    print("\n[WARNING] External dependencies remain after pruning:")
    for x in sorted(unresolved):
        print(" ", x)

    print(
        "\nIf input_tensor appears here, the raw output path still depends on the "
        "original input branch. Verify whether that tensor is only used by "
        "shape/true_image_shape branches."
    )
    raise RuntimeError("Unresolved external inputs remain after pruning.")

# 4. Keep only required initializers.
old_initializers = list(graph.initializer)
kept_initializers = [
    copy.deepcopy(init) for init in old_initializers if init.name in required_tensors
]

print("Original initializer count:", len(old_initializers))
print("Kept initializer count:", len(kept_initializers))

# 5. Rebuild graph inputs.
new_input_vi = helper.make_tensor_value_info(
    new_input_name,
    TensorProto.FLOAT,
    [1, 3, 300, 300],
)

# 6. Rebuild graph outputs.
# Manually specify raw output shapes to avoid dynamic shape side effects.
new_outputs = [
    helper.make_tensor_value_info(
        "raw_detection_boxes",
        TensorProto.FLOAT,
        [1, 1917, 4],
    ),
    helper.make_tensor_value_info(
        "raw_detection_scores",
        TensorProto.FLOAT,
        [1, 1917, 91],
    ),
]

# 7. Optional: keep only relevant value_info entries.
old_value_infos = list(graph.value_info)
needed_value_info_names = (
    set(required_tensors) | produced_names | set(output_names) | {new_input_name}
)

kept_value_infos = [
    copy.deepcopy(vi) for vi in old_value_infos if vi.name in needed_value_info_names
]

# 8. Write changes back to the graph.
del graph.node[:]
graph.node.extend(kept_nodes)

del graph.initializer[:]
graph.initializer.extend(kept_initializers)

del graph.input[:]
graph.input.append(new_input_vi)

del graph.output[:]
graph.output.extend(new_outputs)

del graph.value_info[:]
graph.value_info.extend(kept_value_infos)

# 9. Validate and save.
onnx.checker.check_model(model)
onnx.save(model, dst_model)

print("\nSaved:", dst_model)

# 10. Print the new model inputs and outputs.
core = onnx.load(dst_model)

print("\nInputs:")
for inp in core.graph.input:
    t = inp.type.tensor_type
    shape = []
    for d in t.shape.dim:
        shape.append(d.dim_value if d.dim_value else d.dim_param)
    print(" ", inp.name, "dtype:", t.elem_type, "shape:", shape)

print("\nOutputs:")
for out in core.graph.output:
    t = out.type.tensor_type
    shape = []
    for d in t.shape.dim:
        shape.append(d.dim_value if d.dim_value else d.dim_param)
    print(" ", out.name, "dtype:", t.elem_type, "shape:", shape)
