# YOLO26n demo

This demo follows the same command style as the YOLOv5s/YOLOv8n demos:

```text
image -> preprocess -> input bin -> Neuron RuntimeV2 -> 6 raw output bins -> YOLO26 postprocess -> result image/json
```

## Host-side Conversion

Input size: `512x512`  
Cut ONNX output: `yolo26n_6head.onnx`

```bash
cd examples/yolo26n/convert_model
conda activate yolo-export
yolo export model=yolo26n format=onnx opset=13 imgsz=512

conda activate np8-cp310
python cut_onnx.py
cd ../../..
python prepare_calibration_data.py path=./datasets/coco128/images/train2017 imgsz=512
cd examples/yolo26n/convert_model
python convert_mtk_fp32.py
python convert_mtk_int8.py
```

Outputs:
- `yolo26n_mtk_fp32.tflite`
- `yolo26n_mtk_int8.tflite`

## Models

- `model/int8/`
  - `yolo26n_int8.dla`
  - `yolo26n_mtk_int8.tflite`
- `model/fp32/`
  - `yolo26n_fp32.dla`
  - `yolo26n_mtk_fp32.tflite`

## Model Conversion

After copying the host-converted TFLite files to the target board, convert them in place:

```bash
cd model/int8
ncc-tflite --arch=mdla2.0 -d yolo26n_int8.dla yolo26n_mtk_int8.tflite

cd ../fp32
ncc-tflite --arch=mdla2.0 -d yolo26n_fp32.dla yolo26n_mtk_fp32.tflite --relax-fp32
```

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Run

Run with the default INT8 model on both default images (`bus.jpg` + `zidane.jpg`):

```bash
./build/yolo26n_demo
```

Run with the FP32 model:

```bash
./build/yolo26n_demo --fp32
```

Run with one or more specific images:

```bash
./build/yolo26n_demo \
  --image assets/images/bus.jpg \
  --image assets/images/zidane.jpg
```

Run without `NeuronRuntimeV2_run` (still loads the `.dla`, prints each input/output buffer size in bytes, letterbox-preprocesses images, and checks the buffer matches `input[0]`):

```bash
./build/yolo26n_demo --skip-runtime
```

## Generated Artifacts

- `outputs/yolo26n/detections/<image>.json`
- `outputs/yolo26n/vis/<image>_det.png`

## Implementation Notes

- The demo targets `1x3x512x512` NCHW input.
- Model outputs are decoupled 6 heads:
  - bbox logits: `1x4x64x64`, `1x4x32x32`, `1x4x16x16`
  - cls logits: `1x80x64x64`, `1x80x32x32`, `1x80x16x16`
- Bounding box decode uses direct bbox regression (no DFL).
- Class logits use sigmoid; there is no objectness branch and no anchors.
- Final filtering uses confidence threshold + class-wise NMS.
