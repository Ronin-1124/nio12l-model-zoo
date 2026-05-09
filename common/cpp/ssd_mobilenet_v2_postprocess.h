#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/cpp/image_types.h"

namespace mtkdemo {

struct SsdQuantParam {
  float scale = 1.0f;
  int zero_point = 0;
};

/// Decode SSD MobileNet V2 pre-NMS outputs:
/// `raw_detection_boxes` [1,N,4] and `raw_detection_scores` [1,N,91].
/// Assumptions:
///   - boxes are in TensorFlow order **[ymin, xmin, ymax, xmax]**, normalized to [0,1];
///   - scores are already per-class probabilities (score-converted in TF OD graph).
/// The input image is stretch-resized to 300x300, so boxes are mapped back with
/// **x * image_width** and **y * image_height**.
std::vector<Detection> DecodeSsdMobilenetV2Raw(
    const std::vector<uint8_t>& raw_boxes_bytes, const std::vector<uint8_t>& raw_scores_bytes,
    int image_width, int image_height, float score_threshold, float iou_threshold, size_t top_k);

std::vector<Detection> DecodeSsdMobilenetV2RawInt8(
    const std::vector<uint8_t>& raw_boxes_bytes, const std::vector<uint8_t>& raw_scores_bytes,
    const SsdQuantParam& boxes_quant, const SsdQuantParam& scores_quant, int image_width,
    int image_height, float score_threshold, float iou_threshold, size_t top_k);

}  // namespace mtkdemo
