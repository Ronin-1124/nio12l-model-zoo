# YOLO26n-pose demo

This demo follows the same command style as other YOLO pose demos:

```text
image -> preprocess -> Neuron RuntimeV2 -> 9 raw outputs -> decode + NMS -> result image/json
```

## Models

- `models/yolo26n-pose/int8/`
  - `yolo26n-pose_int8.dla`
  - `yolo26n-pose_mtk_int8.tflite`
- `models/yolo26n-pose/fp32/`
  - `yolo26n-pose_fp32.dla`
  - `yolo26n-pose_mtk_fp32.tflite`

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Run

Run INT8 model:

```bash
./build/yolo26n-pose_demo
```

Run FP32 model:

```bash
./build/yolo26n-pose_demo --fp32
```

Run with specific images:

```bash
./build/yolo26n-pose_demo \
  --image assets/images/bus.jpg \
  --image assets/images/zidane.jpg
```

Run without `NeuronRuntimeV2_run` (still loads the `.dla`, prints each input/output buffer size in bytes, letterbox-preprocesses images, and checks the buffer matches `input[0]`):

```bash
./build/yolo26n-pose_demo --skip-runtime
```

## Generated Artifacts

- `outputs/yolo26n-pose/detections/<image>.json`
- `outputs/yolo26n-pose/vis/<image>_pose.png`

## Implementation Notes

- Input is `1x3x512x512` NCHW.
- Nine outputs are grouped as:
  - keypoints heads: `1x51x64x64`, `1x51x32x32`, `1x51x16x16` (`51 = 17 * 3`)
  - bbox direct (ltrb) heads: `1x4x64x64`, `1x4x32x32`, `1x4x16x16`
  - person score heads: `1x1x64x64`, `1x1x32x32`, `1x1x16x16`
- Postprocess does direct bbox decode, person score sigmoid, keypoint decode, confidence filter and NMS.
