#include "common/cpp/detection_utils.h"

#include <algorithm>

namespace mtkdemo {

float IntersectionOverUnion(const Detection& a, const Detection& b) {
  const float inter_x1 = std::max(a.x1, b.x1);
  const float inter_y1 = std::max(a.y1, b.y1);
  const float inter_x2 = std::min(a.x2, b.x2);
  const float inter_y2 = std::min(a.y2, b.y2);
  const float inter_w = std::max(0.0f, inter_x2 - inter_x1);
  const float inter_h = std::max(0.0f, inter_y2 - inter_y1);
  const float inter_area = inter_w * inter_h;
  const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  const float denom = area_a + area_b - inter_area;
  return denom > 0.0f ? inter_area / denom : 0.0f;
}

std::vector<Detection> NonMaximumSuppressionPerClass(std::vector<Detection> detections,
                                                      float iou_threshold) {
  std::sort(detections.begin(), detections.end(), [](const Detection& lhs, const Detection& rhs) {
    return lhs.confidence > rhs.confidence;
  });

  std::vector<Detection> kept;
  std::vector<bool> suppressed(detections.size(), false);
  for (size_t i = 0; i < detections.size(); ++i) {
    if (suppressed[i]) {
      continue;
    }
    kept.push_back(detections[i]);
    for (size_t j = i + 1; j < detections.size(); ++j) {
      if (suppressed[j] || detections[i].class_id != detections[j].class_id) {
        continue;
      }
      if (IntersectionOverUnion(detections[i], detections[j]) > iou_threshold) {
        suppressed[j] = true;
      }
    }
  }
  return kept;
}

std::vector<Detection> LimitTopKDetections(std::vector<Detection> detections, size_t top_k) {
  if (top_k == 0 || detections.size() <= top_k) {
    return detections;
  }
  std::partial_sort(
      detections.begin(), detections.begin() + static_cast<std::ptrdiff_t>(top_k), detections.end(),
      [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });
  detections.resize(top_k);
  return detections;
}

}  // namespace mtkdemo
