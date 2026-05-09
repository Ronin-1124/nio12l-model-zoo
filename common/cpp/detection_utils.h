#pragma once

#include <cstddef>
#include <vector>

#include "common/cpp/image_types.h"

namespace mtkdemo {

float IntersectionOverUnion(const Detection& a, const Detection& b);

std::vector<Detection> NonMaximumSuppressionPerClass(std::vector<Detection> detections,
                                                      float iou_threshold);

std::vector<Detection> LimitTopKDetections(std::vector<Detection> detections, size_t top_k);

}  // namespace mtkdemo
