import os
import mtk_converter

_here = os.path.dirname(os.path.abspath(__file__))

converter = mtk_converter.OnnxConverter.from_model_proto_file(
    model_proto_file=os.path.join(_here, "yolo11n-obb_9head.onnx"),
    input_names=["images"],
    input_shapes=[(1, 3, 1024, 1024)],
)

converter.decompose_batched_matmul_ops = True
converter.convert_to_tflite(
    output_file=os.path.join(_here, "..", "model", "fp32", "yolo11n-obb_mtk_fp32.tflite"),
    tflite_op_export_spec="npsdk_v6",
)
