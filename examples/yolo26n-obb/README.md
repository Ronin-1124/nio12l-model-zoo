# YOLO26n-OBB demo

This demo follows the same command style as other YOLO demos:

```text
image -> preprocess -> Neuron RuntimeV2 -> 9 raw outputs -> OBB decode (no DFL) + rotated NMS -> result image/json
```

## Models

- `models/yolo26n-obb/int8/`
  - `yolo26n-obb_int8.dla`
  - `yolo26n-obb_mtk_int8.tflite`
- `models/yolo26n-obb/fp32/`
  - `yolo26n-obb_fp32.dla`
  - `yolo26n-obb_mtk_fp32.tflite`

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Run

Run INT8 model:

```bash
./build/yolo26n-obb_demo
```

Run FP32 model:

```bash
./build/yolo26n-obb_demo --fp32
```

Default image (DOTA-style aerial scene):

```bash
./build/yolo26n-obb_demo
# default image: assets/images/boats.jpg
```

Run with specific images:

```bash
./build/yolo26n-obb_demo \
  --image assets/images/boats.jpg
```

Run without `NeuronRuntimeV2_run` (still loads the `.dla`, prints each input/output buffer size in bytes, letterbox-preprocesses images, and checks the buffer matches `input[0]`):

```bash
./build/yolo26n-obb_demo --skip-runtime
```

## Generated Artifacts

- `outputs/yolo26n-obb/detections/<image>.json`
- `outputs/yolo26n-obb/vis/<image>_obb.png`

## Implementation Notes

- Input is `1x3x1024x1024` NCHW.
- Nine outputs come from the `one2one` head and are grouped as:
  - angle heads: `1x1x128x128`, `1x1x64x64`, `1x1x32x32`
  - bbox heads:  `1x4x128x128`, `1x4x64x64`, `1x4x32x32`  (direct ltrb regression, no DFL)
  - class heads: `1x15x128x128`, `1x15x64x64`, `1x15x32x32`
- Postprocess does direct bbox decode `(l + r) * stride / scale` for size and rotated
  center adjustment `cx_in += ((r-l)/2)*stride * cos(theta) - ((b-t)/2)*stride * sin(theta)`,
  class sigmoid, direct angle decode `angle=raw_angle`, confidence filter, then
  class-wise rotated NMS based on polygon intersection.
- In INT8 mode, tensors are dequantized with per-output `(scale, zero_point)` before
  decoding. For this model, all three class heads have `zero_point=127` and large
  scales, so INT8 class scores tend to cluster around 0.5; FP32 is more reliable for
  accuracy evaluation.
