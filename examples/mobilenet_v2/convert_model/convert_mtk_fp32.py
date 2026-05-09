import os
import tensorflow as tf
import mtk_converter

_here = os.path.dirname(os.path.abspath(__file__))

loaded = tf.saved_model.load(os.path.join(_here, "model"))
infer = loaded.signatures["serving_default"]


@tf.function
def model(x):
    return infer(inputs=x)


cf = model.get_concrete_function(
    tf.TensorSpec(shape=(1, 224, 224, 3), dtype=tf.float32)
)

converter = mtk_converter.TensorFlowConverter.from_concrete_functions(
    concrete_functions=[cf],
    input_names=["x"],
    input_shapes=[(1, 224, 224, 3)],
    output_names=["Identity"],
)

converter.convert_to_tflite(
    output_file=os.path.join(_here, "..", "model", "fp32", "mobilenet_v2_mtk_fp32.tflite"),
    tflite_op_export_spec="npsdk_v6",
)
