import os
import numpy as np
import mtk_converter

_here = os.path.dirname(os.path.abspath(__file__))
_onnx = os.path.join(_here, "resnet50.onnx")
_calib_dir = os.path.join(_here, "..", "..", "..", "calibration_dataset", "224")
_out = os.path.join(_here, "..", "model", "int8", "resnet50_mtk_int8.tflite")

converter = mtk_converter.OnnxConverter.from_model_proto_file(
    _onnx,
    input_names=["input_1"],
    input_shapes=[(1, 224, 224, 3)],
    output_names=["activation_49"],
)


def data_gen():
    """Calibration data: NCHW float32 [0,1] -> NHWC float32 [0,1]."""
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
converter.convert_to_tflite(output_file=_out, tflite_op_export_spec="npsdk_v6")
