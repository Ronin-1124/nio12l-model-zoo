from onnx import utils

utils.extract_model(
    input_path="yolo26n-seg.onnx",
    output_path="yolo26n-seg_10head.onnx",
    input_names=["images"],
    output_names=[
        "/model.23/one2one_cv4.0/one2one_cv4.0.2/Conv_output_0",
        "/model.23/one2one_cv4.1/one2one_cv4.1.2/Conv_output_0",
        "/model.23/one2one_cv4.2/one2one_cv4.2.2/Conv_output_0",
        "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0",
        "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0",
        "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0",
        "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0",
        "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0",
        "/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0",
        "output1",
    ],
)
