import onnx

model = onnx.load("yolo11n-seg.onnx")
model = onnx.shape_inference.infer_shapes(model)
onnx.save(model, "yolo11n-seg.onnx")
