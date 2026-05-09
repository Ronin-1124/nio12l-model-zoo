import os
import mtk_converter

_here = os.path.dirname(os.path.abspath(__file__))

converter = mtk_converter.OnnxConverter.from_model_proto_file(
    model_proto_file=os.path.join(_here, "yolov8n-pose_9head.onnx"),
    input_names=["images"],
    input_shapes=[(1, 3, 640, 640)],
)

converter.convert_to_tflite(
    output_file=os.path.join(_here, "..", "model", "fp32", "yolov8n-pose_mtk_fp32.tflite"),
    tflite_op_export_spec="npsdk_v6",
)
