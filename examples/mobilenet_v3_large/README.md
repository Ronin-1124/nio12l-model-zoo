# MobileNetV3 Large demo

This demo follows the same command style as other ImageNet classification examples:

```text
image -> resize RGB NHWC 224x224 -> Neuron RuntimeV2 -> 1001-class logits -> softmax top-k -> json
```

## Host-side Conversion

This example uses TensorFlow SavedModel downloaded by `download_model.py`.

```bash
cd examples/mobilenet_v3_large/convert_model
conda activate np8-cp310
python download_model.py

cd ../../..
python prepare_calibration_data.py path=./datasets/imagenet100 imgsz=224

cd examples/mobilenet_v3_large/convert_model
python convert_mtk_fp32.py
python convert_mtk_int8.py
```

Note: calibration files are generated in NCHW and transposed to NHWC in the script.

## Models

- `model/int8/`
  - `mobilenet_v3_large_int8.dla`
  - `mobilenet_v3_large_mtk_int8.tflite`
- `model/fp32/`
  - `mobilenet_v3_large_fp32.dla`
  - `mobilenet_v3_large_mtk_fp32.tflite`

## Model Conversion

After copying the host-converted TFLite files to the target board, convert them in place:

```bash
cd model/int8
ncc-tflite --arch=mdla2.0 -d mobilenet_v3_large_int8.dla mobilenet_v3_large_mtk_int8.tflite

cd ../fp32
ncc-tflite --arch=mdla2.0 -d mobilenet_v3_large_fp32.dla mobilenet_v3_large_mtk_fp32.tflite --relax-fp32
```

## Run

```bash
./build/mobilenet_v3_large_demo
./build/mobilenet_v3_large_demo --fp32
```

Default images:

```text
assets/images/pickup.jpg
assets/images/rhodesian_ridgeback.jpg
```

Output:

```text
outputs/mobilenet_v3_large/classifications/<image>.json
```

## Implementation Notes

- Input is `1x224x224x3` NHWC RGB.
- Output has 1001 classes. `ImagenetLabels()[0]` is `background`.
- INT8 input uses `scale=0.00392157, zero_point=-128`.
- INT8 output uses `scale=0.063437, zero_point=-62`.
