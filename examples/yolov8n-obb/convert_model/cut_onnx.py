from onnx import utils

utils.extract_model(
    input_path="yolov8n-obb.onnx",
    output_path="yolov8n-obb_9head.onnx",
    input_names=["images"],
    output_names=[
        "/model.22/cv4.0/cv4.0.2/Conv_output_0",
        "/model.22/cv4.1/cv4.1.2/Conv_output_0",
        "/model.22/cv4.2/cv4.2.2/Conv_output_0",
        "/model.22/cv3.0/cv3.0.2/Conv_output_0",
        "/model.22/cv3.1/cv3.1.2/Conv_output_0",
        "/model.22/cv3.2/cv3.2.2/Conv_output_0",
        "/model.22/cv2.0/cv2.0.2/Conv_output_0",
        "/model.22/cv2.1/cv2.1.2/Conv_output_0",
        "/model.22/cv2.2/cv2.2.2/Conv_output_0",
    ],
)
