# mtk-genio-1200-model-zoo

Model zoo and runtime examples for MediaTek Genio 1200 using Neuron SDK

This repository is intended to collect:

- model conversion notes
- Neuron RuntimeV2 API examples
- preprocessing and postprocessing implementations
- classification and detection demos

## Target platform

- MediaTek Genio 1200
- Official Ubuntu image
- Neuron SDK

## Environment dependencies

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

## Expected tools:

- ncc-tflite
- neuronrt
- runtime_api_sample

## Expected headers:

- /usr/include/neuron/api

## Check runtime library:

```bash
ls -l /usr/lib/libneuronusdk_runtime.mtk.so*
```

## Build

Use CMake to build all demo executables:

```bash
cmake -S . -B build
cmake --build build -j
```

## Example docs

For model-specific usage, commands, and caveats, see the README inside each example folder:

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

