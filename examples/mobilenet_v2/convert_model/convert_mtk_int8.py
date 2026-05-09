import os
import numpy as np
import tensorflow as tf
import mtk_converter

_here = os.path.dirname(os.path.abspath(__file__))
_calib_dir = os.path.join(_here, "..", "..", "..", "calibration_dataset", "224")

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


def data_gen():
    if not os.path.isdir(_calib_dir):
        raise FileNotFoundError(
            f"Calibration directory not found: {_calib_dir}. "
            "Run prepare_calibration_data.py with imgsz=224 first."
        )
    for fn in sorted(os.listdir(_calib_dir)):
        im = np.load(os.path.join(_calib_dir, fn))
        yield [im.transpose(0, 2, 3, 1)]  # NCHW → NHWC


converter.quantize = True
converter.calibration_data_gen = data_gen
converter.convert_to_tflite(
    output_file=os.path.join(_here, "..", "model", "int8", "mobilenet_v2_mtk_int8.tflite"),
    tflite_op_export_spec="npsdk_v6",
)
