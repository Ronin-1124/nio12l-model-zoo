from onnx import utils

utils.extract_model(
    input_path="yolov8n.onnx",
    output_path="yolov8n_3head.onnx",
    input_names=["images"],
    output_names=[
        "/model.22/Concat_output_0",
        "/model.22/Concat_1_output_0",
        "/model.22/Concat_2_output_0",
    ],
)
