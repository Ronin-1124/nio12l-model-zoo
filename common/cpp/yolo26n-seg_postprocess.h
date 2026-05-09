#pragma once

#include <cstdint>
#include <vector>

#include "common/cpp/yolov8n-seg_postprocess.h"

namespace mtkdemo {

// YOLO26n-seg: 10 outputs (3 coeff + 3 bbox(ltrb direct) + 3 cls logits + 1 proto).
std::vector<SegmentationInstance> DecodeYolo26SegMultiOutput(
    const std::vector<std::vector<uint8_t>>& coeff_outputs,
    const std::vector<std::vector<uint8_t>>& bbox_outputs,
    const std::vector<std::vector<uint8_t>>& cls_outputs, const std::vector<uint8_t>& proto_output,
    const LetterboxInfo& info, int image_width, int image_height, float confidence_threshold,
    float iou_threshold, float mask_threshold, int class_count, int input_size = 512,
    int proto_h = 128, int proto_w = 128);

std::vector<SegmentationInstance> DecodeYolo26SegMultiOutputInt8(
    const std::vector<std::vector<uint8_t>>& coeff_outputs,
    const std::vector<YoloV8SegQuantParam>& coeff_quant,
    const std::vector<std::vector<uint8_t>>& bbox_outputs,
    const std::vector<YoloV8SegQuantParam>& bbox_quant,
    const std::vector<std::vector<uint8_t>>& cls_outputs,
    const std::vector<YoloV8SegQuantParam>& cls_quant, const std::vector<uint8_t>& proto_output,
    const YoloV8SegQuantParam& proto_quant, const LetterboxInfo& info, int image_width,
    int image_height, float confidence_threshold, float iou_threshold, float mask_threshold,
    int class_count, int input_size = 512, int proto_h = 128, int proto_w = 128);

}  // namespace mtkdemo
