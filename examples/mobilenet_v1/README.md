# MobileNetV1 demo

This demo follows the same command style as other examples:

```text
image -> resize RGB NHWC 224x224 -> Neuron RuntimeV2 -> 1001-class logits -> softmax top-k -> json
```

## Models

- `models/mobilenet_v1/int8/`
  - `mobilenet_v1_int8.dla`
  - `mobilenet_v1_1.0_224_mtk_int8.tflite`
- `models/mobilenet_v1/fp32/`
  - `mobilenet_v1_fp32.dla`
  - `mobilenet_v1_1.0_224_mtk_fp32.tflite`

## Model Conversion

After copying the host-converted TFLite files to the target board, convert them in place:

```bash
cd models/mobilenet_v1/int8
ncc-tflite --arch=mdla2.0 -d mobilenet_v1_int8.dla mobilenet_v1_1.0_224_mtk_int8.tflite

cd ../fp32
ncc-tflite --arch=mdla2.0 -d mobilenet_v1_fp32.dla mobilenet_v1_1.0_224_mtk_fp32.tflite --relax-fp32
```

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Run

Run INT8 model:

```bash
./build/mobilenet_v1_demo
```

Run FP32 model:

```bash
./build/mobilenet_v1_demo --fp32
```

Default images:

```text
assets/images/pickup.jpg
assets/images/rhodesian_ridgeback.jpg
```

Run with a specific image:

```bash
./build/mobilenet_v1_demo --image assets/images/pickup.jpg
```

Run without `NeuronRuntimeV2_run` (still loads the `.dla`, prints each input/output buffer size in bytes, preprocesses images, and checks the buffer matches `input[0]`):

```bash
./build/mobilenet_v1_demo --skip-runtime
```

## Generated Artifacts

- `outputs/mobilenet_v1/classifications/<image>.json`

## Implementation Notes

- Input is `1x224x224x3` NHWC RGB.
- Pixel values are normalized to `[0, 1]`.
- INT8 input uses `scale=0.00392157, zero_point=-128`.
- Output has 1001 classes. `ImagenetLabels()[0]` is `background`; ImageNet classes are at indexes `1..1000`.
- The demo dequantizes INT8 output logits with `scale=0.139915, zero_point=-60`, applies softmax, then writes top-k classes (default k=5, i.e. top-5).
