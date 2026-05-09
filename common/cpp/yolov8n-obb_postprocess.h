#pragma once

#include <cstdint>
#include <vector>

#include "common/cpp/image_types.h"

namespace mtkdemo {

struct YoloObbQuantParam {
  float scale = 1.0f;
  int zero_point = 0;
};

struct OrientedDetection {
  float cx = 0.0f;
  float cy = 0.0f;
  float w = 0.0f;
  float h = 0.0f;
  float angle = 0.0f;  // Radians.
  float confidence = 0.0f;
  int class_id = -1;
};

std::vector<OrientedDetection> DecodeYoloV8ObbMultiOutput(
    const std::vector<std::vector<uint8_t>>& angle_outputs,
    const std::vector<std::vector<uint8_t>>& cls_outputs,
    const std::vector<std::vector<uint8_t>>& bbox_outputs, const LetterboxInfo& info,
    int image_width, int image_height, float confidence_threshold, float iou_threshold,
    int class_count = 15, int reg_max = 16, int input_size = 1024);

std::vector<OrientedDetection> DecodeYoloV8ObbMultiOutputInt8(
    const std::vector<std::vector<uint8_t>>& angle_outputs,
    const std::vector<YoloObbQuantParam>& angle_quant,
    const std::vector<std::vector<uint8_t>>& cls_outputs,
    const std::vector<YoloObbQuantParam>& cls_quant,
    const std::vector<std::vector<uint8_t>>& bbox_outputs,
    const std::vector<YoloObbQuantParam>& bbox_quant, const LetterboxInfo& info, int image_width,
    int image_height, float confidence_threshold, float iou_threshold, int class_count = 15,
    int reg_max = 16, int input_size = 1024);

}  // namespace mtkdemo

