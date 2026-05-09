# YOLOv8n-pose demo

This demo follows the same command style as other YOLO demos:

```text
image -> preprocess -> Neuron RuntimeV2 -> 9 raw outputs -> decode + NMS -> result image/json
```

## Models

- `models/yolov8n-pose/int8/`
  - `yolov8n-pose_int8.dla`
  - `yolov8n-pose_mtk_int8.tflite`
- `models/yolov8n-pose/fp32/`
  - `yolov8n-pose_fp32.dla`
  - `yolov8n-pose_mtk_fp32.tflite`

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Run

Run INT8 model:

```bash
./build/yolov8n-pose_demo
```

Run FP32 model:

```bash
./build/yolov8n-pose_demo --fp32
```

Run with specific images:

```bash
./build/yolov8n-pose_demo \
  --image assets/images/bus.jpg \
  --image assets/images/zidane.jpg
```

Run without `NeuronRuntimeV2_run` (still loads the `.dla`, prints each input/output buffer size in bytes, letterbox-preprocesses images, and checks the buffer matches `input[0]`):

```bash
./build/yolov8n-pose_demo --skip-runtime
```

## Generated Artifacts

- `outputs/yolov8n-pose/detections/<image>.json`
- `outputs/yolov8n-pose/vis/<image>_pose.png`

## Implementation Notes

- Input is `1x3x640x640` NCHW.
- Nine outputs are grouped as:
  - keypoints heads: `1x51x80x80`, `1x51x40x40`, `1x51x20x20` (`51 = 17 * 3`)
  - person score heads: `1x1x80x80`, `1x1x40x40`, `1x1x20x20`
  - bbox DFL heads: `1x64x80x80`, `1x64x40x40`, `1x64x20x20` (`64 = 4 * 16`, `reg_max=16`)
- Postprocess does DFL decode, person score sigmoid, keypoint decode, confidence filter and NMS.
