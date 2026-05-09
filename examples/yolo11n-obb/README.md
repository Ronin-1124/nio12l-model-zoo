# YOLO11n-OBB demo

This demo follows the same command style as other YOLO demos:

```text
image -> preprocess -> Neuron RuntimeV2 -> 9 raw outputs -> OBB decode + rotated NMS -> result image/json
```

## Models

- `models/yolo11n-obb/int8/`
  - `yolo11n-obb_int8.dla`
  - `yolo11n-obb_mtk_int8.tflite`
- `models/yolo11n-obb/fp32/`
  - `yolo11n-obb_fp32.dla`
  - `yolo11n-obb_mtk_fp32.tflite`

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

Build only this demo:

```bash
cmake --build build --target yolo11n-obb_demo -j
```

## Run

Run INT8 model:

```bash
./build/yolo11n-obb_demo
```

Run FP32 model:

```bash
./build/yolo11n-obb_demo --fp32
```

Default image (DOTA-style aerial scene):

```bash
./build/yolo11n-obb_demo
# default image: assets/images/boats.jpg
```

Run with specific images:

```bash
./build/yolo11n-obb_demo \
  --image assets/images/boats.jpg
```

Run without `NeuronRuntimeV2_run` (still loads the `.dla`, prints each input/output buffer size in bytes, letterbox-preprocesses images, and checks the buffer matches `input[0]`):

```bash
./build/yolo11n-obb_demo --skip-runtime
```

## Generated Artifacts

- `outputs/yolo11n-obb/detections/<image>.json`
- `outputs/yolo11n-obb/vis/<image>_obb.png`

## Implementation Notes

- Input is `1x3x1024x1024` NCHW.
- Nine outputs are grouped as:
  - angle heads: `1x1x128x128`, `1x1x64x64`, `1x1x32x32`
  - class heads: `1x15x128x128`, `1x15x64x64`, `1x15x32x32`
  - bbox DFL heads: `1x64x128x128`, `1x64x64x64`, `1x64x32x32`
- Postprocess does DFL decode (`reg_max=16`), rotated center decode, class sigmoid, angle decode
  `angle=(sigmoid(raw)-0.25)*pi`, confidence filter, then class-wise rotated NMS.
- In INT8 mode, tensors are dequantized with per-output `(scale, zero_point)` before DFL/sigmoid/angle decode.
