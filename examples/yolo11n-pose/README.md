# YOLO11n-pose demo

This demo follows the same command style as other YOLO pose demos:

```text
image -> preprocess -> Neuron RuntimeV2 -> 9 raw outputs -> decode + NMS -> result image/json
```

## Models

- `models/yolo11n-pose/int8/`
  - `yolo11n-pose_int8.dla`
  - `yolo11n-pose_mtk_int8.tflite`
- `models/yolo11n-pose/fp32/`
  - `yolo11n-pose_fp32.dla`
  - `yolo11n-pose_mtk_fp32.tflite`

## Model Conversion

After copying the host-converted TFLite files to the target board, convert them in place:

```bash
cd models/yolo11n-pose/int8
ncc-tflite --arch=mdla2.0 -d yolo11n-pose_int8.dla yolo11n-pose_mtk_int8.tflite

cd ../fp32
ncc-tflite --arch=mdla2.0 -d yolo11n-pose_fp32.dla yolo11n-pose_mtk_fp32.tflite --relax-fp32
```

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Run

Run INT8 model:

```bash
./build/yolo11n-pose_demo
```

Run FP32 model:

```bash
./build/yolo11n-pose_demo --fp32
```

Run with specific images:

```bash
./build/yolo11n-pose_demo \
  --image assets/images/bus.jpg \
  --image assets/images/zidane.jpg
```

Run without `NeuronRuntimeV2_run` (still loads the `.dla`, prints each input/output buffer size in bytes, letterbox-preprocesses images, and checks the buffer matches `input[0]`):

```bash
./build/yolo11n-pose_demo --skip-runtime
```

## Generated Artifacts

- `outputs/yolo11n-pose/detections/<image>.json`
- `outputs/yolo11n-pose/vis/<image>_pose.png`

## Implementation Notes

- Input is `1x3x512x512` NCHW.
- Nine outputs are grouped as:
  - keypoints heads: `1x51x64x64`, `1x51x32x32`, `1x51x16x16` (`51 = 17 * 3`)
  - person score heads: `1x1x64x64`, `1x1x32x32`, `1x1x16x16`
  - bbox DFL heads: `1x64x64x64`, `1x64x32x32`, `1x64x16x16` (`64 = 4 * 16`, `reg_max=16`)
- Postprocess does DFL decode, person score sigmoid, keypoint decode, confidence filter and NMS.
