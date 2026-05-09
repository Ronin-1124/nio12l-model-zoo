import onnx

model = onnx.load("yolo26n-seg.onnx")
model = onnx.shape_inference.infer_shapes(model)
onnx.save(model, "yolo26n-seg.onnx")
