# YOLO12n demo

This demo follows the same command style as the YOLOv5s/YOLOv8n demos:

```text
image -> preprocess -> input bin -> Neuron RuntimeV2 -> 6 raw output bins -> YOLO12 postprocess -> result image/json
```

## Models

- `models/yolo12n/int8/`
  - `yolo12n_int8.dla`
  - `yolo12n_mtk_int8.tflite`
- `models/yolo12n/fp32/`
  - `yolo12n_fp32.dla`
  - `yolo12n_mtk_fp32.tflite`

## Model Conversion

After copying the host-converted TFLite files to the target board, convert them in place:

```bash
cd models/yolo12n/int8
ncc-tflite --arch=mdla2.0 -d yolo12n_int8.dla yolo12n_mtk_int8.tflite

cd ../fp32
ncc-tflite --arch=mdla2.0 -d yolo12n_fp32.dla yolo12n_mtk_fp32.tflite --relax-fp32
```

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Run

Run with the default INT8 model on both default images (`bus.jpg` + `zidane.jpg`):

```bash
./build/yolo12n_demo
```

Run with the FP32 model:

```bash
./build/yolo12n_demo --fp32
```

Run with one or more specific images:

```bash
./build/yolo12n_demo \
  --image assets/images/bus.jpg \
  --image assets/images/zidane.jpg
```

Run without `NeuronRuntimeV2_run` (still loads the `.dla`, prints each input/output buffer size in bytes, letterbox-preprocesses images, and checks the buffer matches `input[0]`):

```bash
./build/yolo12n_demo --skip-runtime
```

## Generated Artifacts

- `outputs/yolo12n/detections/<image>.json`
- `outputs/yolo12n/vis/<image>_det.png`

## Implementation Notes

- The demo targets `1x3x512x512` NCHW input.
- Model outputs are decoupled 6 heads:
  - bbox logits: `1x64x64x64`, `1x64x32x32`, `1x64x16x16`
  - cls logits: `1x80x64x64`, `1x80x32x32`, `1x80x16x16`
- Bounding box decode uses DFL (`reg_max=16`) + grid/stride decode.
- Class logits use sigmoid; there is no objectness branch and no anchors.
- Final filtering uses confidence threshold + class-wise NMS.
