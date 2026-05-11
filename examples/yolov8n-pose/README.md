# YOLOv8n-pose demo

This demo follows the same command style as other YOLO demos:

```text
image -> preprocess -> Neuron RuntimeV2 -> 9 raw outputs -> decode + NMS -> result image/json
```

## Host-side Conversion

Input size: `640x640`  
Cut ONNX output: `yolov8n-pose_9head.onnx`

```bash
cd examples/yolov8n-pose/convert_model
conda activate yolo-export
yolo export model=yolov8n-pose format=onnx opset=13 imgsz=640

conda activate np8-cp310
python cut_onnx.py
cd ../../..
python prepare_calibration_data.py path=./datasets/coco128/images/train2017 imgsz=640
cd examples/yolov8n-pose/convert_model
python convert_mtk_fp32.py
python convert_mtk_int8.py
```

## Models

- `model/int8/`
  - `yolov8n-pose_int8.dla`
  - `yolov8n-pose_mtk_int8.tflite`
- `model/fp32/`
  - `yolov8n-pose_fp32.dla`
  - `yolov8n-pose_mtk_fp32.tflite`

## Model Conversion

After copying the host-converted TFLite files to the target board, convert them in place:

```bash
cd model/int8
ncc-tflite --arch=mdla2.0 -d yolov8n-pose_int8.dla yolov8n-pose_mtk_int8.tflite

cd ../fp32
ncc-tflite --arch=mdla2.0 -d yolov8n-pose_fp32.dla yolov8n-pose_mtk_fp32.tflite --relax-fp32
```

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
