import onnx

model = onnx.load("yolo11n-obb.onnx")
model = onnx.shape_inference.infer_shapes(model)
onnx.save(model, "yolo11n-obb.onnx")
