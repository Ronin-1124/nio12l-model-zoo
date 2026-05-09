#pragma once

#include <cstdint>
#include <vector>

#include "common/cpp/image_types.h"
#include "common/cpp/yolov8n-obb_postprocess.h"

namespace mtkdemo {

// YOLO26n-OBB has 9 raw heads with the following per-scale layout:
//   angle: 1 channel
//   bbox:  4 channels (l, t, r, b directly regressed, no DFL)
//   cls:   class_count channels (logits, sigmoid required)
// The detection struct and quant param struct are reused from
// yolov8n-obb_postprocess.h.

std::vector<OrientedDetection> DecodeYolo26ObbMultiOutput(
    const std::vector<std::vector<uint8_t>>& angle_outputs,
    const std::vector<std::vector<uint8_t>>& bbox_outputs,
    const std::vector<std::vector<uint8_t>>& cls_outputs, const LetterboxInfo& info,
    int image_width, int image_height, float confidence_threshold, float iou_threshold,
    int class_count = 15, int input_size = 1024);

std::vector<OrientedDetection> DecodeYolo26ObbMultiOutputInt8(
    const std::vector<std::vector<uint8_t>>& angle_outputs,
    const std::vector<YoloObbQuantParam>& angle_quant,
    const std::vector<std::vector<uint8_t>>& bbox_outputs,
    const std::vector<YoloObbQuantParam>& bbox_quant,
    const std::vector<std::vector<uint8_t>>& cls_outputs,
    const std::vector<YoloObbQuantParam>& cls_quant, const LetterboxInfo& info, int image_width,
    int image_height, float confidence_threshold, float iou_threshold, int class_count = 15,
    int input_size = 1024);

}  // namespace mtkdemo
