"""EfficientNet-B0 → MTK FP32 TFLite (via ONNX)."""

import os
import mtk_converter

_here = os.path.dirname(os.path.abspath(__file__))
_onnx = os.path.join(_here, "efficient_b0.onnx")
_out = os.path.join(_here, "..", "model", "fp32", "efficientnet_b0_mtk_fp32.tflite")

converter = mtk_converter.OnnxConverter.from_model_proto_file(
    _onnx,
    input_names=["image_input"],
    input_shapes=[(1, 224, 224, 3)],
    output_names=["probs"],
)

converter.rewrite_batchnorm_to_dwconv_ops = False
converter.convert_to_tflite(output_file=_out, tflite_op_export_spec="npsdk_v6")
