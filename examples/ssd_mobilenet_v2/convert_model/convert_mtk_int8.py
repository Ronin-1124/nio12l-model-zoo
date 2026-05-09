"""SSD MobileNet V2 core → MTK INT8 TFLite (via ONNX)."""

import os
import numpy as np
import mtk_converter

_here = os.path.dirname(os.path.abspath(__file__))
_onnx = os.path.join(_here, "ssd_mobilenet_v2_core.onnx")
_calib_dir = os.path.join(_here, "..", "..", "..", "calibration_dataset", "300")
_out = os.path.join(_here, "..", "model", "int8", "ssd_mobilenet_v2_mtk_int8.tflite")

converter = mtk_converter.OnnxConverter.from_model_proto_file(
    _onnx,
    input_names=["preprocessed_input"],
    input_shapes=[(1, 3, 300, 300)],
    output_names=["raw_detection_boxes", "raw_detection_scores"],
)

converter.rewrite_batchnorm_to_dwconv_ops = False


def data_gen():
    """Calibration data must match model input: NCHW float32 [0,1] @ 300x300."""
    if not os.path.isdir(_calib_dir):
        raise FileNotFoundError(
            f"Calibration directory not found: {_calib_dir}. "
            "Run prepare_calibration_data.py with imgsz=300 first."
        )
    for fn in sorted(os.listdir(_calib_dir)):
        im = np.load(os.path.join(_calib_dir, fn))
        # prepare_calibration_data outputs NCHW float32 [0,1]
        yield [im]


converter.quantize = True
converter.calibration_data_gen = data_gen
converter.convert_to_tflite(output_file=_out, tflite_op_export_spec="npsdk_v6")
