import onnx

model = onnx.load("yolo12n.onnx")
model = onnx.shape_inference.infer_shapes(model)
onnx.save(model, "yolo12n.onnx")
