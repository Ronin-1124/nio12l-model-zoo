#pragma once

#include <cstdint>
#include <vector>

#include "common/cpp/image_types.h"

namespace mtkdemo {

struct YoloV8QuantParam {
  float scale = 1.0f;
  int zero_point = 0;
};

std::vector<Detection> DecodeYoloV8MultiOutput(const std::vector<std::vector<uint8_t>>& outputs,
                                               const LetterboxInfo& info, int image_width,
                                               int image_height, float confidence_threshold,
                                               float iou_threshold, int class_count,
                                               int reg_max = 16);

std::vector<Detection> DecodeYoloV8MultiOutputInt8(
    const std::vector<std::vector<uint8_t>>& outputs,
    const std::vector<YoloV8QuantParam>& quant_params, const LetterboxInfo& info, int image_width,
    int image_height, float confidence_threshold, float iou_threshold, int class_count,
    int reg_max = 16);

}  // namespace mtkdemo
