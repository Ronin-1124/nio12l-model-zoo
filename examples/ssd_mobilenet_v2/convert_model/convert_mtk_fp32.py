"""SSD MobileNet V2 core → MTK FP32 TFLite (via ONNX)."""

import os
import mtk_converter

_here = os.path.dirname(os.path.abspath(__file__))
_onnx = os.path.join(_here, "ssd_mobilenet_v2_core.onnx")
_out = os.path.join(_here, "..", "model", "fp32", "ssd_mobilenet_v2_mtk_fp32.tflite")

converter = mtk_converter.OnnxConverter.from_model_proto_file(
    _onnx,
    input_names=["preprocessed_input"],
    input_shapes=[(1, 3, 300, 300)],
    output_names=["raw_detection_boxes", "raw_detection_scores"],
)

converter.rewrite_batchnorm_to_dwconv_ops = False
converter.convert_to_tflite(output_file=_out, tflite_op_export_spec="npsdk_v6")
