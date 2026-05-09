#pragma once

#include <cstdint>
#include <vector>

#include "common/cpp/image_types.h"

namespace mtkdemo {

struct Yolo26QuantParam {
  float scale = 1.0f;
  int zero_point = 0;
};

std::vector<Detection> DecodeYolo26MultiOutput(
    const std::vector<std::vector<uint8_t>>& bbox_outputs,
    const std::vector<std::vector<uint8_t>>& cls_outputs, const LetterboxInfo& info,
    int image_width, int image_height, float confidence_threshold, float iou_threshold,
    int class_count, int input_size = 512);

std::vector<Detection> DecodeYolo26MultiOutputInt8(
    const std::vector<std::vector<uint8_t>>& bbox_outputs,
    const std::vector<Yolo26QuantParam>& bbox_quant_params,
    const std::vector<std::vector<uint8_t>>& cls_outputs,
    const std::vector<Yolo26QuantParam>& cls_quant_params, const LetterboxInfo& info,
    int image_width, int image_height, float confidence_threshold, float iou_threshold,
    int class_count, int input_size = 512);

}  // namespace mtkdemo
