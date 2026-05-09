#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/cpp/image_types.h"

namespace mtkdemo {

struct TensorQuantParam {
  float scale = 1.0f;
  int zero_point = 0;
};

/// Decode YOLOv5 detections from multiple output tensors (multi-scale detection heads).
/// Each output tensor is decoded independently, then all detections are merged into one list
/// and a single NMS pass is applied.
/// This is the main entry point for models that produce separate tensors per scale
/// (e.g. YOLOv5s with three scales: 80x80, 40x40, 20x20).
std::vector<Detection> DecodeYoloV5MultiOutput(
    const std::vector<std::vector<uint8_t>>& outputs, const LetterboxInfo& info, int image_width,
    int image_height, float confidence_threshold, float iou_threshold, int class_count);

/// Decode YOLOv5 detections from int8 output tensors with per-output quantization parameters.
std::vector<Detection> DecodeYoloV5MultiOutputInt8(
    const std::vector<std::vector<uint8_t>>& outputs,
    const std::vector<TensorQuantParam>& output_quant_params, const LetterboxInfo& info,
    int image_width, int image_height, float confidence_threshold, float iou_threshold,
    int class_count);

}  // namespace mtkdemo
