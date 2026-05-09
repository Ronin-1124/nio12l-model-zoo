# SSD MobileNet V2 demo

This demo targets the trimmed SSD MobileNet V2 graph where in-graph preprocessing and NMS were removed. Preprocessing and postprocessing are implemented externally in C++.

## Preprocessing (different from original SavedModel uint8 NHWC raw image input)

- Read image with `LoadImage` (stb, RGB output). If your source is OpenCV BGR, swap R/B first.
- Stretch-resize to **300x300** (no letterbox), bilinear interpolation.
- Normalize to **[0, 1]** with `x/255` for the current TFLite/DLA build (`preprocessed_input`: `scale=1/255, zero_point=-128`).
- Layout is **NCHW** (`1x3x300x300`).

INT8 path quantizes this `[0, 1]` float tensor with `kInt8InputQuant` in `ssd_mobilenet_v2_config.h`.

## Outputs and postprocessing

- `raw_detection_boxes`: `[1,1917,4]`, interpreted as **[ymin, xmin, ymax, xmax]**, normalized to [0, 1].
- `raw_detection_scores`: `[1,1917,91]`, including background at index 0. Scores are already class probabilities (TF OD score-converted, per-class sigmoid).
- External postprocessing: background removal, score threshold, per-class NMS, global top-k.
- Box mapping back to original image uses stretch geometry:
  - `x_orig = x_norm * W_orig`
  - `y_orig = y_norm * H_orig`

If your export differs in box order or score semantics, adjust `common/cpp/ssd_mobilenet_v2_postprocess.cpp` accordingly. Thresholds and top-k are configured in `examples/ssd_mobilenet_v2/ssd_mobilenet_v2_config.h`.

> SSD confidence distribution is typically flatter than YOLO because each anchor uses independent per-class sigmoid scores. Negative anchors may retain medium scores (e.g. 0.3~0.5), so this demo uses a higher default threshold (`kScoreThreshold=0.5`) than many YOLO examples.

## Labels

`common/cpp/coco91_labels.cpp` embeds TensorFlow `mscoco_complete_label_map.pbtxt` `display_name` entries. Index `0..90` aligns with the 91-dim score vector, including placeholder ids (e.g. `"12"`, `"26"`) as defined by the official map.

## Host-side Conversion

This example expects `ssd_mobilenet_v2.onnx`, then exports a core ONNX for MTK conversion.

```bash
cd examples/ssd_mobilenet_v2/convert_model
conda activate np8-cp310
python download_model.py
# Prepare ssd_mobilenet_v2.onnx in this directory if it is not already present.
python export_onnx.py

cd ../..
python prepare_calibration_data.py path=./datasets/coco128/images/train2017 imgsz=300

cd examples/ssd_mobilenet_v2/convert_model
python convert_mtk_fp32.py
python convert_mtk_int8.py
```

Notes:
- `export_onnx.py` generates `ssd_mobilenet_v2_core.onnx`.
- INT8 calibration must match `300x300` input.

## Model Conversion

After copying the host-converted TFLite files to the target board, convert them in place:

```bash
cd model/int8
ncc-tflite --arch=mdla2.0 -d ssd_mobilenet_v2_int8.dla ssd_mobilenet_v2_mtk_int8.tflite

cd ../fp32
ncc-tflite --arch=mdla2.0 -d ssd_mobilenet_v2_fp32.dla ssd_mobilenet_v2_mtk_fp32.tflite --relax-fp32
```

## Build and run

```bash
cmake -S . -B build
cmake --build build -j
./build/ssd_mobilenet_v2_demo
./build/ssd_mobilenet_v2_demo --fp32 --image assets/images/bus.jpg
./build/ssd_mobilenet_v2_demo --skip-runtime
```

Default models:

- INT8: `model/int8/ssd_mobilenet_v2_int8.dla`
- FP32: `model/fp32/ssd_mobilenet_v2_fp32.dla`

Default output directory: `outputs/ssd_mobilenet_v2/` (`vis/` for visualization, `detections/` for JSON).

## Output tensor pairing

Neuron output order may differ from TFLite name order. The demo validates tensor pairing by checking element count consistency against expected shapes (`N*4` for boxes and `N*91` for scores with matching `N`) instead of relying only on buffer size.

## Runtime notes

Same as other demos: install Neuron runtime packages on target board. If device enumeration fails (e.g. `neuronrt -d`), use `ncc-tflite` and `--skip-runtime` for preprocessing and I/O-size sanity checks.
