import onnx

model = onnx.load("yolov10n.onnx")
model = onnx.shape_inference.infer_shapes(model)
onnx.save(model, "yolov10n.onnx")
