# MobileNetV2 demo

This demo follows the same command style as other ImageNet classification examples:

```text
image -> resize RGB NHWC 224x224 -> Neuron RuntimeV2 -> 1001-class logits -> softmax top-k -> json
```

## Models

- `models/mobilenet_v2/int8/`
  - `mobilenet_v2_int8.dla`
  - `mobilenet_v2_mtk_int8.tflite`
- `models/mobilenet_v2/fp32/`
  - `mobilenet_v2_fp32.dla`
  - `mobilenet_v2_mtk_fp32.tflite`

## Run

```bash
./build/mobilenet_v2_demo
./build/mobilenet_v2_demo --fp32
```

Default images:

```text
assets/images/pickup.jpg
assets/images/rhodesian_ridgeback.jpg
```

Output:

```text
outputs/mobilenet_v2/classifications/<image>.json
```

## Implementation Notes

- Input is `1x224x224x3` NHWC RGB.
- Output has 1001 classes. `ImagenetLabels()[0]` is `background`.
- INT8 input uses `scale=0.00392157, zero_point=-128`.
- INT8 output uses `scale=0.0671408, zero_point=-66`.
