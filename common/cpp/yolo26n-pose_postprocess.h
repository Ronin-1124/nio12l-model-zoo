#pragma once

#include <cstdint>
#include <vector>

#include "common/cpp/image_types.h"
#include "common/cpp/yolov8n-pose_postprocess.h"

namespace mtkdemo {

std::vector<PoseInstance> DecodeYolo26PoseMultiOutput(
    const std::vector<std::vector<uint8_t>>& kpt_outputs,
    const std::vector<std::vector<uint8_t>>& score_outputs,
    const std::vector<std::vector<uint8_t>>& bbox_outputs, const LetterboxInfo& info,
    int image_width, int image_height, float confidence_threshold, float iou_threshold,
    int keypoint_count = 17, int input_size = 512);

std::vector<PoseInstance> DecodeYolo26PoseMultiOutputInt8(
    const std::vector<std::vector<uint8_t>>& kpt_outputs,
    const std::vector<YoloV8PoseQuantParam>& kpt_quant,
    const std::vector<std::vector<uint8_t>>& score_outputs,
    const std::vector<YoloV8PoseQuantParam>& score_quant,
    const std::vector<std::vector<uint8_t>>& bbox_outputs,
    const std::vector<YoloV8PoseQuantParam>& bbox_quant, const LetterboxInfo& info,
    int image_width, int image_height, float confidence_threshold, float iou_threshold,
    int keypoint_count = 17, int input_size = 512);

}  // namespace mtkdemo

