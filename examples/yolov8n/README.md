# YOLOv8n demo

This demo follows the same command style as the YOLOv5s demo:

```text
image -> preprocess -> input bin -> Neuron RuntimeV2 -> raw output bin -> YOLOv8 postprocess -> result image/json
```

## Host-side Conversion

Input size: `640x640`  
Cut ONNX output: `yolov8n_3head.onnx`

```bash
cd examples/yolov8n/convert_model
conda activate yolo-export
yolo export model=yolov8n format=onnx opset=13 imgsz=640

conda activate np8-cp310
python cut_onnx.py
cd ../../..
python prepare_calibration_data.py path=./datasets/coco128/images/train2017 imgsz=640
cd examples/yolov8n/convert_model
python convert_mtk_fp32.py
python convert_mtk_int8.py
```

Outputs:
- `yolov8n_mtk_fp32.tflite`
- `yolov8n_mtk_int8.tflite`

## Models

- `model/int8/`
  - `yolov8n_int8.dla`
  - `yolov8n_mtk_int8.tflite`
- `model/fp32/`
  - `yolov8n_fp32.dla`
  - `yolov8n_mtk_fp32.tflite`

## Model Conversion

After copying the host-converted TFLite files to the target board, convert them in place:

```bash
cd model/int8
ncc-tflite --arch=mdla2.0 -d yolov8n_int8.dla yolov8n_mtk_int8.tflite

cd ../fp32
ncc-tflite --arch=mdla2.0 -d yolov8n_fp32.dla yolov8n_mtk_fp32.tflite --relax-fp32
```

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Run

Run with the default INT8 model on both default images (`bus.jpg` + `zidane.jpg`):

```bash
./build/yolov8n_demo
```

Run with the FP32 model:

```bash
./build/yolov8n_demo --fp32
```

Run with one or more specific images:

```bash
./build/yolov8n_demo \
  --image assets/images/bus.jpg \
  --image assets/images/zidane.jpg
```

Run without `NeuronRuntimeV2_run` (still loads the `.dla`, prints each input/output buffer size in bytes, letterbox-preprocesses images, and checks the buffer matches `input[0]`):

```bash
./build/yolov8n_demo --skip-runtime
```

## Generated Artifacts

- `outputs/yolov8n/detections/<image>.json`
- `outputs/yolov8n/vis/<image>_det.png`

## Implementation Notes

- The demo supports `1x3x640x640` NCHW input for both FP32 and INT8 models.
- YOLOv8n uses one output tensor with shape `1x1x84x8400`.
- INT8 mode uses quantized input and dequantizes the single output tensor before postprocess.
- The postprocess expects YOLOv8-style rows: 4 box values followed by 80 class scores.

## Current Caveat

The runtime path builds and executes, but the current exported YOLOv8n model outputs should still be
validated against the host export pipeline. In the current board smoke test, INT8 produced no
post-threshold detections, while FP32 produced saturated class scores. If this happens, re-check the
exported TFLite graph, output tensor semantics, and quantization parameters before treating the
visualized boxes as model-quality results.
