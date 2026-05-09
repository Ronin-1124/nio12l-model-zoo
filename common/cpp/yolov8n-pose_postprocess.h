#pragma once

#include <cstdint>
#include <vector>

#include "common/cpp/image_types.h"

namespace mtkdemo {

struct YoloV8PoseQuantParam {
  float scale = 1.0f;
  int zero_point = 0;
};

struct PoseInstance {
  Detection detection;
  // COCO17 keypoints, each as {x, y, score} in original image coordinates.
  std::vector<float> keypoints;  // size = keypoint_count * 3
};

std::vector<PoseInstance> DecodeYoloV8PoseMultiOutput(
    const std::vector<std::vector<uint8_t>>& kpt_outputs,
    const std::vector<std::vector<uint8_t>>& score_outputs,
    const std::vector<std::vector<uint8_t>>& bbox_outputs, const LetterboxInfo& info,
    int image_width, int image_height, float confidence_threshold, float iou_threshold,
    int keypoint_count = 17, int reg_max = 16, int input_size = 640);

std::vector<PoseInstance> DecodeYoloV8PoseMultiOutputInt8(
    const std::vector<std::vector<uint8_t>>& kpt_outputs,
    const std::vector<YoloV8PoseQuantParam>& kpt_quant,
    const std::vector<std::vector<uint8_t>>& score_outputs,
    const std::vector<YoloV8PoseQuantParam>& score_quant,
    const std::vector<std::vector<uint8_t>>& bbox_outputs,
    const std::vector<YoloV8PoseQuantParam>& bbox_quant, const LetterboxInfo& info,
    int image_width, int image_height, float confidence_threshold, float iou_threshold,
    int keypoint_count = 17, int reg_max = 16, int input_size = 640);

}  // namespace mtkdemo

