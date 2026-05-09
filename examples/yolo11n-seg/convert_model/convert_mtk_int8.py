import os
import numpy as np
import mtk_converter

_here = os.path.dirname(os.path.abspath(__file__))
_calib_dir = os.path.join(_here, "..", "..", "..", "calibration_dataset", "512")

converter = mtk_converter.OnnxConverter.from_model_proto_file(
    model_proto_file=os.path.join(_here, "yolo11n-seg_10head.onnx"),
    input_names=["images"],
    input_shapes=[(1, 3, 512, 512)],
)


def data_gen():
    """Return an iterator for the calibration dataset."""
    if not os.path.isdir(_calib_dir):
        raise FileNotFoundError(
            f"Calibration directory not found: {_calib_dir}. "
            "Run prepare_calibration_data.py with the matching imgsz first."
        )
    for fn in sorted(os.listdir(_calib_dir)):
        yield [np.load(os.path.join(_calib_dir, fn))]


converter.quantize = True
converter.calibration_data_gen = data_gen
converter.decompose_batched_matmul_ops = True
converter.convert_to_tflite(
    output_file=os.path.join(_here, "..", "model", "int8", "yolo11n-seg_mtk_int8.tflite"),
    tflite_op_export_spec="npsdk_v6",
)
