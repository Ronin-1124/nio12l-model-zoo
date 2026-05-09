# mtk-genio-1200-model-zoo

Model zoo, conversion pipelines, and runtime examples for MediaTek Genio 1200 using Neuron SDK.

Each example under `examples/` is a self-contained model directory covering the full lifecycle:

- **Host-side conversion** (`convert_model/`) — Python scripts to produce MTK-compatible FP32/INT8 TFLite models
- **Model storage** (`model/`) — compiled `.dla` and `.tflite` files (fp32 / int8)
- **Board-side inference** (`main.cpp`, `*_config.h`) — C++ Neuron RuntimeV2 demos

## Host-side setup (model conversion)

Two Conda environments are used:

- `np8-cp310`: main conversion environment (MTK converter, TensorFlow/ONNX tooling)
- `yolo-export`: YOLO ONNX export environment (latest `ultralytics`)

Use `yolo-export` only for exporting YOLO ONNX models. Use `np8-cp310` for all MTK conversion scripts.

### Conversion workflow

1. **Prepare source model** — YOLO: export ONNX via `yolo-export`; Non-YOLO: use `download_model.py` or provide ONNX
2. **Generate calibration data** (for INT8) — run `prepare_calibration_data.py` from repo root
3. **Convert** — run `convert_mtk_fp32.py` and `convert_mtk_int8.py` in each model's `convert_model/` directory

See per-model READMEs under `examples/` for exact commands.

### Shared resources

- `prepare_calibration_data.py` — generates `calibration_dataset/<imgsz>/batch-*.npy`
- `datasets/` — sample image sets for calibration (coco128, dota128, imagenet100)
- `requirements.txt` — pinned Python dependencies for the `np8-cp310` environment

## Board-side setup (inference)

### Target platform

- MediaTek Genio 1200
- Official Ubuntu image
- Neuron SDK

### Environment dependencies

Install basic build tools:

```bash
sudo apt update
sudo apt install gcc g++ cmake
```

Install Neuron runtime packages (required on target board):

```bash
sudo apt install mediatek-libneuron mediatek-libneuron-dev mediatek-neuron-utils
```

Install OpenCV development package if you use OpenCV-based image pipelines/scripts in your workflow:

```bash
sudo apt install libopencv-dev
```

Note: the current C++ demos in this repository use `stb` for image I/O and do not directly link to OpenCV.

### Expected tools

- ncc-tflite
- neuronrt
- runtime_api_sample

### Expected headers

- /usr/include/neuron/api

### Check runtime library

```bash
ls -l /usr/lib/libneuronusdk_runtime.mtk.so*
```

## Build

Use CMake to build all demo executables:

```bash
cmake -S . -B build
cmake --build build -j
```

## Delivering models to board

After host-side conversion, copy the generated `.tflite` files from `examples/<model>/model/{fp32,int8}/` to the board, then use `ncc-tflite` to compile them to `.dla` format. See each model's README for the exact `ncc-tflite` command.

## Example docs

For model-specific usage (conversion + inference), commands, and caveats, see the README inside each example folder:

- `examples/mobilenet_v1/README.md`
- `examples/mobilenet_v2/README.md`
- `examples/mobilenet_v3_large/README.md`
- `examples/mobilenet_v3_small/README.md`
- `examples/efficientnet_b0_classification/README.md`
- `examples/resnet50/README.md`
- `examples/ssd_mobilenet_v2/README.md`
- `examples/yolov5s/README.md`
- `examples/yolov8n/README.md`
- `examples/yolov8n-obb/README.md`
- `examples/yolo11n-obb/README.md`
- `examples/yolo26n-obb/README.md`
- `examples/yolov8n-pose/README.md`
- `examples/yolo11n-pose/README.md`
- `examples/yolo26n-pose/README.md`
- `examples/yolo11n/README.md`
- `examples/yolov10n/README.md`
- `examples/yolo26n/README.md`
- `examples/yolo12n/README.md`
- `examples/yolov8n-seg/README.md`
- `examples/yolo11n-seg/README.md`
- `examples/yolo26n-seg/README.md`
