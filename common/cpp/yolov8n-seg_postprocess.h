#pragma once

#include <cstdint>
#include <vector>

#include "common/cpp/image_types.h"

namespace mtkdemo {

struct YoloV8SegQuantParam {
  float scale = 1.0f;
  int zero_point = 0;
};

struct SegmentationInstance {
  Detection detection;
  std::vector<uint8_t> mask;
};

std::vector<SegmentationInstance> DecodeYoloV8SegMultiOutput(
    const std::vector<std::vector<uint8_t>>& coeff_outputs,
    const std::vector<std::vector<uint8_t>>& cls_outputs,
    const std::vector<std::vector<uint8_t>>& bbox_outputs, const std::vector<uint8_t>& proto_output,
    const LetterboxInfo& info, int image_width, int image_height, float confidence_threshold,
    float iou_threshold, float mask_threshold, int class_count, int reg_max = 16,
    int input_size = 640, int proto_h = 160, int proto_w = 160);

std::vector<SegmentationInstance> DecodeYoloV8SegMultiOutputInt8(
    const std::vector<std::vector<uint8_t>>& coeff_outputs,
    const std::vector<YoloV8SegQuantParam>& coeff_quant,
    const std::vector<std::vector<uint8_t>>& cls_outputs,
    const std::vector<YoloV8SegQuantParam>& cls_quant,
    const std::vector<std::vector<uint8_t>>& bbox_outputs,
    const std::vector<YoloV8SegQuantParam>& bbox_quant, const std::vector<uint8_t>& proto_output,
    const YoloV8SegQuantParam& proto_quant, const LetterboxInfo& info, int image_width,
    int image_height, float confidence_threshold, float iou_threshold, float mask_threshold,
    int class_count, int reg_max = 16, int input_size = 640, int proto_h = 160, int proto_w = 160);

}  // namespace mtkdemo
