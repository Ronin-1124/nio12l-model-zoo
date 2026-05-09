# MobileNetV3 Small demo

This demo follows the same command style as other ImageNet classification examples:

```text
image -> resize RGB NHWC 224x224 -> Neuron RuntimeV2 -> 1001-class logits -> softmax top-k -> json
```

## Models

- `models/mobilenet_v3_small/int8/`
  - `mobilenet_v3_small_int8.dla`
  - `mobilenet_v3_small_mtk_int8.tflite`
- `models/mobilenet_v3_small/fp32/`
  - `mobilenet_v3_small_fp32.dla`
  - `mobilenet_v3_small_mtk_fp32.tflite`

## Model Conversion

After copying the host-converted TFLite files to the target board, convert them in place:

```bash
cd models/mobilenet_v3_small/int8
ncc-tflite --arch=mdla2.0 -d mobilenet_v3_small_int8.dla mobilenet_v3_small_mtk_int8.tflite

cd ../fp32
ncc-tflite --arch=mdla2.0 -d mobilenet_v3_small_fp32.dla mobilenet_v3_small_mtk_fp32.tflite --relax-fp32
```

## Run

```bash
./build/mobilenet_v3_small_demo
./build/mobilenet_v3_small_demo --fp32
```

Default images:

```text
assets/images/pickup.jpg
assets/images/rhodesian_ridgeback.jpg
```

Output:

```text
outputs/mobilenet_v3_small/classifications/<image>.json
```

## Implementation Notes

- Input is `1x224x224x3` NHWC RGB.
- Output has 1001 classes. `ImagenetLabels()[0]` is `background`.
- INT8 input uses `scale=0.00392157, zero_point=-128`.
- INT8 output uses `scale=0.0760708, zero_point=-62`.
