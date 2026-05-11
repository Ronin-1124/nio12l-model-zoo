# ResNet50 demo

This demo follows the same command style as the other ImageNet classification examples:

```text
image -> resize RGB NHWC 224x224 (x/255) -> Neuron RuntimeV2 -> 1001-dim probabilities -> top-k -> json
```

## Host-side Conversion

This example converts an ONNX model named `resnet50.onnx`.

```bash
cd examples/resnet50/convert_model
conda activate np8-cp310
python download_model.py
# Prepare resnet50.onnx in this directory if it is not already present.

cd ../../..
python prepare_calibration_data.py path=./datasets/imagenet100 imgsz=224

cd examples/resnet50/convert_model
python convert_mtk_fp32.py
python convert_mtk_int8.py
```

Note: INT8 calibration data is transposed from NCHW to NHWC in `convert_mtk_int8.py`.

## Models

- `model/int8/`
  - `resnet50_int8.dla`
  - `resnet50_mtk_int8.tflite`
- `model/fp32/`
  - `resnet50_fp32.dla`
  - `resnet50_mtk_fp32.tflite`

## Model Conversion

After copying the host-converted TFLite files to the target board, convert them in place:

```bash
cd model/int8
ncc-tflite --arch=mdla2.0 -d resnet50_int8.dla resnet50_mtk_int8.tflite

cd ../fp32
ncc-tflite --arch=mdla2.0 -d resnet50_fp32.dla resnet50_mtk_fp32.tflite --relax-fp32
```

## Run

```bash
./build/resnet50_demo
./build/resnet50_demo --fp32
```

Default images:

```text
assets/images/pickup.jpg
assets/images/rhodesian_ridgeback.jpg
```

Output:

```text
outputs/resnet50/classifications/<image>.json
```

## Implementation Notes

- Input `input_1`: `{1, 224, 224, 3}` NHWC RGB, normalized with `x/255` (INT8 `scale=1/255, zero_point=-128`, real-value range `[0, 1]`). If mean/std normalization is required by training, it is already folded into model weights and should not be repeated externally.
- Output `activation_49`: `{1, 1001}`, already softmax probabilities (INT8 `scale=1/256, zero_point=-128`, range `[0, 1]`). `RunClassificationDemo` sets `output_is_probabilities=true` and skips external softmax.
- The model uses 0-based ImageNet-1000 class ids, but output tensor size is 1001 where the last element is dummy. The demo reads `output_tensor_size=1001`, then truncates to `class_count=1000` before top-k.
- Labels reuse shared `imagenet_labels.cpp` (1001 entries with `idx 0 = background`) and map class id `i` to `labels[i + 1]` via `label_offset=1`.
- INT8 input quant: `scale=0.00392157, zero_point=-128`.
- INT8 output quant: `scale=0.00390625, zero_point=-128`.
