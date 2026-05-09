import onnx

model = onnx.load("yolo26n-pose.onnx")
model = onnx.shape_inference.infer_shapes(model)
onnx.save(model, "yolo26n-pose.onnx")
