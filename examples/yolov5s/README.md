# YOLOv5s demo

This repository now includes a minimal end-to-end C++ demo for:

```text
image -> preprocess -> input bin -> Neuron RuntimeV2 -> raw output bin -> YOLO postprocess -> result image/json
```

## Directory layout

- `assets/images/`
  - shared sample images such as `bus.jpg` and `zidane.jpg`
- `models/yolov5s/fp32/`
  - `yolov5s_fp32.dla`
  - `yolov5s_mtk_fp32.tflite`
- `models/yolov5s/int8/`
  - `yolov5s_int8.dla`
  - `yolov5s_mtk_int8.tflite`
- `common/cpp/`
  - reusable image I/O, preprocess, runtime, and postprocess helpers
- `examples/yolov5s/`
  - demo entrypoint and model-specific config
- `outputs/yolov5s/`
  - detections and visualization outputs

## Model Conversion

After copying the host-converted TFLite files to the target board, convert them in place:

```bash
cd models/yolov5s/int8
ncc-tflite --arch=mdla2.0 -d yolov5s_int8.dla yolov5s_mtk_int8.tflite

cd ../fp32
ncc-tflite --arch=mdla2.0 -d yolov5s_fp32.dla yolov5s_mtk_fp32.tflite --relax-fp32
```

## Build

The project uses a top-level `CMakeLists.txt`:

```bash
cmake -S . -B build
cmake --build build -j
```

If `cmake` is not installed on the target board, install it first:

```bash
sudo apt install cmake
```

## Run

Run with the default INT8 model on both default images (`bus.jpg` + `zidane.jpg`):

```bash
./build/yolov5s_demo
```

Run with the FP32 model on both default images:

```bash
./build/yolov5s_demo --fp32
```

Run with one or more specific images (repeat `--image`):

```bash
./build/yolov5s_demo \
  --image assets/images/bus.jpg \
  --image assets/images/zidane.jpg
```

Run with a specific model and output directory:

```bash
./build/yolov5s_demo \
  --model models/yolov5s/int8/yolov5s_int8.dla \
  --image assets/images/bus.jpg \
  --output-dir outputs/yolov5s
```

Run without `NeuronRuntimeV2_run` (still loads the `.dla`, prints each input/output buffer size in bytes, letterbox-preprocesses images, and checks the buffer matches `input[0]`):

```bash
./build/yolov5s_demo --skip-runtime
```

Run with runtime options or custom thread/backlog settings:

```bash
./build/yolov5s_demo \
  --runtime-options "some_option_string" \
  --threads 2 --backlog 4
```

## Runtime setup checklist

The board must have MediaTek Neuron runtime packages installed before inference can run.

```bash
sudo apt install mediatek-libneuron mediatek-libneuron-dev mediatek-neuron-utils
```

Verify headers and libraries:

```bash
ls -d /usr/include/neuron/api
ls -l /usr/lib/libneuronusdk_runtime.mtk.so*
```

Verify APUSYS device node access:

```bash
ls -l /dev/apusys*
```

If the device nodes are missing or permissions are too strict, add a udev rule:

```bash
sudo tee /etc/udev/rules.d/99-mtk-apusys.rules > /dev/null <<'RULE'
# MediaTek APUSYS device permissions for Neuron SDK
KERNEL=="apusys*", MODE="0666"
RULE

sudo udevadm control --reload-rules
sudo udevadm trigger
```

If `NeuronRuntimeV2_create(...)` fails with messages such as `Can't open config file` or
`Runtime loadNetworkFromFile fails`, the board runtime environment is not yet ready.

## Generated artifacts

The demo writes:

- `outputs/yolov5s/detections/<image>.json`
- `outputs/yolov5s/vis/<image>_det.png`

## Current implementation notes

- The demo supports both FP32 and INT8 YOLOv5s models with the same `1x3x640x640` NCHW shape.
- INT8 mode is default and uses quantized input/output parameters from the converted model.
- Preprocess uses letterbox resize and NCHW layout (`float32` for FP32 model, quantized `int8` for INT8 model).
- Postprocess decodes three YOLOv5 output heads (`80x80`, `40x40`, `20x20`).
- Result drawing shows category labels and confidence text on the image.

## Current runtime caveat on this board

The source builds with `g++`, but the current runtime smoke test on this machine failed during `NeuronRuntimeV2_create(...)` with runtime-side errors such as:

```text
Can't open config file
ERROR: Cannot prepare execution.
ERROR: Runtime loadNetworkFromFile fails.
```

So the code path is implemented and buildable, but successful inference still depends on the board runtime environment being ready for this DLA model.
