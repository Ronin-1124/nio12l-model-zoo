# EfficientNet-B0 demo

This demo follows the same command style as the other ImageNet classification examples:

```text
image -> resize RGB NHWC 224x224 (x/255) -> Neuron RuntimeV2 -> 1000-class probabilities -> top-k -> json
```

## Models

- `models/efficientnet_b0_classification/int8/`
  - `efficientnet_b0_int8.dla`
  - `efficientnet_b0_mtk_int8.tflite`
- `models/efficientnet_b0_classification/fp32/`
  - `efficientnet_b0_fp32.dla`
  - `efficientnet_b0_mtk_fp32.tflite`

## Run

```bash
./build/efficientnet_b0_classification_demo
./build/efficientnet_b0_classification_demo --fp32
```

Default images:

```text
assets/images/pickup.jpg
assets/images/rhodesian_ridgeback.jpg
```

Output:

```text
outputs/efficientnet_b0_classification/classifications/<image>.json
```

## Implementation Notes

- Input: `{1, 224, 224, 3}` NHWC RGB, normalized with `x/255` (INT8 `image_input` uses `scale=1/255, zero_point=-128`, real-value range `[0, 1]`). If mean/std normalization is required by training, it is already folded into model weights and should not be repeated externally.
- Output: `{1, 1000}` `probs`, already softmax probabilities (INT8 `scale=1/256, zero_point=-128`, range `[0, 1]`). `RunClassificationDemo` sets `output_is_probabilities=true` and skips external softmax.
- Class count is 1000 (no background), unlike TF-SLIM 1001-class MobileNet. The shared `imagenet_labels.cpp` has 1001 entries (`idx 0 = background`), so this demo maps class id `i` to `labels[i + 1]` via `label_offset=1`.
- INT8 input quant: `scale=0.00392157, zero_point=-128`.
- INT8 output quant: `scale=0.00390625, zero_point=-128`.
