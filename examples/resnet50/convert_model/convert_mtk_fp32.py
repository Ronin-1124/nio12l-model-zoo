import os
import mtk_converter

_here = os.path.dirname(os.path.abspath(__file__))
_onnx = os.path.join(_here, "resnet50.onnx")
_out = os.path.join(_here, "..", "model", "fp32", "resnet50_mtk_fp32.tflite")

converter = mtk_converter.OnnxConverter.from_model_proto_file(
    _onnx,
    input_names=["input_1"],
    input_shapes=[(1, 224, 224, 3)],
    output_names=["activation_49"],
)

converter.convert_to_tflite(output_file=_out, tflite_op_export_spec="npsdk_v6")
