# YOLO11n demo

This demo follows the same command style as the YOLOv5s/YOLOv8n demos:

```text
image -> preprocess -> input bin -> Neuron RuntimeV2 -> 6 raw output bins -> YOLO11 postprocess -> result image/json
```

## Host-side Conversion

Input size: `1024x1024`  
Cut ONNX output: `yolo11n_6head.onnx`

```bash
cd examples/yolo11n/convert_model
conda activate yolo-export
yolo export model=yolo11n format=onnx opset=13 imgsz=1024

conda activate np8-cp310
python cut_onnx.py
cd ../../..
python prepare_calibration_data.py path=./datasets/coco128/images/train2017 imgsz=1024
cd examples/yolo11n/convert_model
python convert_mtk_fp32.py
python convert_mtk_int8.py
```

Outputs:
- `yolo11n_mtk_fp32.tflite`
- `yolo11n_mtk_int8.tflite`

## Models

- `model/int8/`
  - `yolo11n_int8.dla`
  - `yolo11n_mtk_int8.tflite`
- `model/fp32/`
  - `yolo11n_fp32.dla`
  - `yolo11n_mtk_fp32.tflite`

## Model Conversion

After copying the host-converted TFLite files to the target board, convert them in place:

```bash
cd model/int8
ncc-tflite --arch=mdla2.0 -d yolo11n_int8.dla yolo11n_mtk_int8.tflite

cd ../fp32
ncc-tflite --arch=mdla2.0 -d yolo11n_fp32.dla yolo11n_mtk_fp32.tflite --relax-fp32
```

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Run

Run with the default INT8 model on both default images (`bus.jpg` + `zidane.jpg`):

```bash
./build/yolo11n_demo
```

Run with the FP32 model:

```bash
./build/yolo11n_demo --fp32
```

Run with one or more specific images:

```bash
./build/yolo11n_demo \
  --image assets/images/bus.jpg \
  --image assets/images/zidane.jpg
```

Run without `NeuronRuntimeV2_run` (still loads the `.dla`, prints each input/output buffer size in bytes, letterbox-preprocesses images, and checks the buffer matches `input[0]`):

```bash
./build/yolo11n_demo --skip-runtime
```

## Generated Artifacts

- `outputs/yolo11n/detections/<image>.json`
- `outputs/yolo11n/vis/<image>_det.png`

## Implementation Notes

- The demo targets `1x3x1024x1024` NCHW input.
- Model outputs are decoupled 6 heads:
  - bbox logits: `1x64x128x128`, `1x64x64x64`, `1x64x32x32`
  - cls logits: `1x80x128x128`, `1x80x64x64`, `1x80x32x32`
- Bounding box decode uses DFL (`reg_max=16`) + grid/stride decode.
- Class logits use sigmoid; there is no objectness branch and no anchors.
- Final filtering uses confidence threshold + class-wise NMS.
