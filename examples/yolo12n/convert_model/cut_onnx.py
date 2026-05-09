from onnx import utils

utils.extract_model(
    input_path="yolo12n.onnx",
    output_path="yolo12n_6head.onnx",
    input_names=["images"],
    output_names=[
        "/model.21/cv2.0/cv2.0.2/Conv_output_0",
        "/model.21/cv2.1/cv2.1.2/Conv_output_0",
        "/model.21/cv2.2/cv2.2.2/Conv_output_0",
        "/model.21/cv3.0/cv3.0.2/Conv_output_0",
        "/model.21/cv3.1/cv3.1.2/Conv_output_0",
        "/model.21/cv3.2/cv3.2.2/Conv_output_0",
    ],
)
