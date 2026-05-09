# YOLOv10n demo

This demo follows the same command style as the YOLOv5s/YOLOv8n demos:

```text
image -> preprocess -> input bin -> Neuron RuntimeV2 -> 6 raw output bins -> YOLOv10 postprocess -> result image/json
```

## Host-side Conversion

Input size: `512x512`  
Cut ONNX output: `yolov10n_6head.onnx`

```bash
cd examples/yolov10n/convert_model
conda activate yolo-export
yolo export model=yolov10n format=onnx opset=13 imgsz=512

conda activate np8-cp310
python cut_onnx.py
python check_yolo_model_list.py
cd ../..
python prepare_calibration_data.py path=./datasets/coco128/images/train2017 imgsz=512
cd examples/yolov10n/convert_model
python convert_mtk_fp32.py
python convert_mtk_int8.py
```

Outputs:
- `yolov10n_mtk_fp32.tflite`
- `yolov10n_mtk_int8.tflite`

## Models

- `model/int8/`
  - `yolov10n_int8.dla`
  - `yolov10n_mtk_int8.tflite`
- `model/fp32/`
  - `yolov10n_fp32.dla`
  - `yolov10n_mtk_fp32.tflite`

## Model Conversion

After copying the host-converted TFLite files to the target board, convert them in place:

```bash
cd model/int8
ncc-tflite --arch=mdla2.0 -d yolov10n_int8.dla yolov10n_mtk_int8.tflite

cd ../fp32
ncc-tflite --arch=mdla2.0 -d yolov10n_fp32.dla yolov10n_mtk_fp32.tflite --relax-fp32
```

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Run

Run with the default INT8 model on both default images (`bus.jpg` + `zidane.jpg`):

```bash
./build/yolov10n_demo
```

Run with the FP32 model:

```bash
./build/yolov10n_demo --fp32
```

Run with one or more specific images:

```bash
./build/yolov10n_demo \
  --image assets/images/bus.jpg \
  --image assets/images/zidane.jpg
```

Run without `NeuronRuntimeV2_run` (still loads the `.dla`, prints each input/output buffer size in bytes, letterbox-preprocesses images, and checks the buffer matches `input[0]`):

```bash
./build/yolov10n_demo --skip-runtime
```

## Generated Artifacts

- `outputs/yolov10n/detections/<image>.json`
- `outputs/yolov10n/vis/<image>_det.png`

## Implementation Notes

- The demo targets `1x3x512x512` NCHW input.
- Model outputs are decoupled 6 heads:
  - bbox logits: `1x64x64x64`, `1x64x32x32`, `1x64x16x16`
  - cls logits: `1x80x64x64`, `1x80x32x32`, `1x80x16x16`
- Bounding box decode uses DFL (`reg_max=16`) + grid/stride decode.
- Class logits use sigmoid; there is no objectness branch and no anchors.
- Final filtering uses confidence threshold + class-wise NMS.
