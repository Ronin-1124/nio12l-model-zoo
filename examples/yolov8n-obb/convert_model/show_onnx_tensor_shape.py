import onnx

model = onnx.load("yolov8n-obb.onnx")
model = onnx.shape_inference.infer_shapes(model)
onnx.save(model, "yolov8n-obb.onnx")
