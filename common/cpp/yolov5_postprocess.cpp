#include "common/cpp/yolov5_postprocess.h"
#include "common/cpp/detection_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace mtkdemo {
namespace {

float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

}  // namespace

std::vector<Detection> DecodeYoloV5MultiOutput(
    const std::vector<std::vector<uint8_t>>& outputs, const LetterboxInfo& info, int image_width,
    int image_height, float confidence_threshold, float iou_threshold, int class_count) {
  constexpr size_t kValuesPerAnchor = 85;  // tx, ty, tw, th, obj + 80 class scores.
  constexpr size_t kAnchors = 3;
  constexpr size_t kChannels = kAnchors * kValuesPerAnchor;
  constexpr int kInputSize = 640;

  static constexpr float kAnchorTable[3][3][2] = {
      {{10.0f, 13.0f}, {16.0f, 30.0f}, {33.0f, 23.0f}},
      {{30.0f, 61.0f}, {62.0f, 45.0f}, {59.0f, 119.0f}},
      {{116.0f, 90.0f}, {156.0f, 198.0f}, {373.0f, 326.0f}},
  };

  std::vector<Detection> all_decoded;

  for (const auto& buf : outputs) {
    if (buf.size() % sizeof(float) != 0) {
      throw std::runtime_error("YOLOv5 output buffer is not float32");
    }

    const float* data = reinterpret_cast<const float*>(buf.data());
    const size_t total_floats = buf.size() / sizeof(float);
    if (total_floats % kChannels != 0) {
      throw std::runtime_error("YOLOv5 multi-output: total elements not divisible by 255");
    }

    const size_t grid_area = total_floats / kChannels;
    const size_t grid_size = static_cast<size_t>(std::sqrt(static_cast<double>(grid_area)));
    if (grid_size * grid_size != grid_area || grid_size == 0) {
      throw std::runtime_error("YOLOv5 multi-output: output grid is not square");
    }

    const int stride = kInputSize / static_cast<int>(grid_size);
    int table_idx = -1;
    if (stride == 8) {
      table_idx = 0;
    } else if (stride == 16) {
      table_idx = 1;
    } else if (stride == 32) {
      table_idx = 2;
    }
    if (table_idx < 0) {
      throw std::runtime_error("Unexpected YOLOv5 grid stride: " + std::to_string(stride));
    }
    const auto& anchors = kAnchorTable[table_idx];

    const size_t plane_size = grid_size * grid_size;
    for (size_t h = 0; h < grid_size; ++h) {
      for (size_t w = 0; w < grid_size; ++w) {
        const size_t spatial_index = h * grid_size + w;

        for (size_t a = 0; a < kAnchors; ++a) {
          const size_t channel_base = a * kValuesPerAnchor;
          const auto value_at = [&](size_t field) {
            return data[(channel_base + field) * plane_size + spatial_index];
          };

          const float tx = value_at(0);
          const float ty = value_at(1);
          const float tw = value_at(2);
          const float th = value_at(3);
          const float objectness = Sigmoid(value_at(4));
          if (objectness < confidence_threshold) {
            continue;
          }

          int best_class = -1;
          float best_class_score = 0.0f;
          for (int c = 0; c < class_count; ++c) {
            const float score = Sigmoid(value_at(5 + static_cast<size_t>(c)));
            if (score > best_class_score) {
              best_class_score = score;
              best_class = c;
            }
          }

          const float confidence = objectness * best_class_score;
          if (confidence < confidence_threshold) {
            continue;
          }

          const float stride_f = static_cast<float>(stride);
          const float cx = (Sigmoid(tx) * 2.0f - 0.5f + static_cast<float>(w)) * stride_f;
          const float cy = (Sigmoid(ty) * 2.0f - 0.5f + static_cast<float>(h)) * stride_f;
          const float bw_norm = Sigmoid(tw) * 2.0f;
          const float bh_norm = Sigmoid(th) * 2.0f;
          const float bw = bw_norm * bw_norm * anchors[a][0];
          const float bh = bh_norm * bh_norm * anchors[a][1];

          float x1 = cx - bw * 0.5f;
          float y1 = cy - bh * 0.5f;
          float x2 = cx + bw * 0.5f;
          float y2 = cy + bh * 0.5f;

          x1 = (x1 - static_cast<float>(info.pad_x)) / info.scale;
          y1 = (y1 - static_cast<float>(info.pad_y)) / info.scale;
          x2 = (x2 - static_cast<float>(info.pad_x)) / info.scale;
          y2 = (y2 - static_cast<float>(info.pad_y)) / info.scale;

          Detection detection;
          detection.x1 = std::clamp(x1, 0.0f, static_cast<float>(image_width - 1));
          detection.y1 = std::clamp(y1, 0.0f, static_cast<float>(image_height - 1));
          detection.x2 = std::clamp(x2, 0.0f, static_cast<float>(image_width - 1));
          detection.y2 = std::clamp(y2, 0.0f, static_cast<float>(image_height - 1));
          detection.confidence = confidence;
          detection.class_id = best_class;
          all_decoded.push_back(detection);
        }
      }
    }
  }

  return NonMaximumSuppressionPerClass(std::move(all_decoded), iou_threshold);
}

std::vector<Detection> DecodeYoloV5MultiOutputInt8(
    const std::vector<std::vector<uint8_t>>& outputs,
    const std::vector<TensorQuantParam>& output_quant_params, const LetterboxInfo& info,
    int image_width, int image_height, float confidence_threshold, float iou_threshold,
    int class_count) {
  constexpr size_t kValuesPerAnchor = 85;  // tx, ty, tw, th, obj + 80 class scores.
  constexpr size_t kAnchors = 3;
  constexpr size_t kChannels = kAnchors * kValuesPerAnchor;
  constexpr int kInputSize = 640;

  if (outputs.size() != output_quant_params.size()) {
    throw std::runtime_error("YOLOv5 int8 decode: outputs and quant params size mismatch");
  }

  static constexpr float kAnchorTable[3][3][2] = {
      {{10.0f, 13.0f}, {16.0f, 30.0f}, {33.0f, 23.0f}},
      {{30.0f, 61.0f}, {62.0f, 45.0f}, {59.0f, 119.0f}},
      {{116.0f, 90.0f}, {156.0f, 198.0f}, {373.0f, 326.0f}},
  };

  std::vector<Detection> all_decoded;

  for (size_t output_idx = 0; output_idx < outputs.size(); ++output_idx) {
    const auto& buf = outputs[output_idx];
    const TensorQuantParam& quant = output_quant_params[output_idx];
    if (quant.scale <= 0.0f) {
      throw std::runtime_error("YOLOv5 int8 decode: quant scale must be positive");
    }
    const size_t total_values = buf.size();
    if (total_values % kChannels != 0) {
      throw std::runtime_error("YOLOv5 int8 decode: total elements not divisible by 255");
    }

    const int8_t* data = reinterpret_cast<const int8_t*>(buf.data());
    const size_t grid_area = total_values / kChannels;
    const size_t grid_size = static_cast<size_t>(std::sqrt(static_cast<double>(grid_area)));
    if (grid_size * grid_size != grid_area || grid_size == 0) {
      throw std::runtime_error("YOLOv5 int8 decode: output grid is not square");
    }

    const int stride = kInputSize / static_cast<int>(grid_size);
    int table_idx = -1;
    if (stride == 8) {
      table_idx = 0;
    } else if (stride == 16) {
      table_idx = 1;
    } else if (stride == 32) {
      table_idx = 2;
    }
    if (table_idx < 0) {
      throw std::runtime_error("Unexpected YOLOv5 grid stride: " + std::to_string(stride));
    }
    const auto& anchors = kAnchorTable[table_idx];

    const size_t plane_size = grid_size * grid_size;
    for (size_t h = 0; h < grid_size; ++h) {
      for (size_t w = 0; w < grid_size; ++w) {
        const size_t spatial_index = h * grid_size + w;

        for (size_t a = 0; a < kAnchors; ++a) {
          const size_t channel_base = a * kValuesPerAnchor;
          const auto value_at = [&](size_t field) {
            const int8_t q = data[(channel_base + field) * plane_size + spatial_index];
            return (static_cast<float>(q) - static_cast<float>(quant.zero_point)) * quant.scale;
          };

          const float tx = value_at(0);
          const float ty = value_at(1);
          const float tw = value_at(2);
          const float th = value_at(3);
          const float objectness = Sigmoid(value_at(4));
          if (objectness < confidence_threshold) {
            continue;
          }

          int best_class = -1;
          float best_class_score = 0.0f;
          for (int c = 0; c < class_count; ++c) {
            const float score = Sigmoid(value_at(5 + static_cast<size_t>(c)));
            if (score > best_class_score) {
              best_class_score = score;
              best_class = c;
            }
          }

          const float confidence = objectness * best_class_score;
          if (confidence < confidence_threshold) {
            continue;
          }

          const float stride_f = static_cast<float>(stride);
          const float cx = (Sigmoid(tx) * 2.0f - 0.5f + static_cast<float>(w)) * stride_f;
          const float cy = (Sigmoid(ty) * 2.0f - 0.5f + static_cast<float>(h)) * stride_f;
          const float bw_norm = Sigmoid(tw) * 2.0f;
          const float bh_norm = Sigmoid(th) * 2.0f;
          const float bw = bw_norm * bw_norm * anchors[a][0];
          const float bh = bh_norm * bh_norm * anchors[a][1];

          float x1 = cx - bw * 0.5f;
          float y1 = cy - bh * 0.5f;
          float x2 = cx + bw * 0.5f;
          float y2 = cy + bh * 0.5f;

          x1 = (x1 - static_cast<float>(info.pad_x)) / info.scale;
          y1 = (y1 - static_cast<float>(info.pad_y)) / info.scale;
          x2 = (x2 - static_cast<float>(info.pad_x)) / info.scale;
          y2 = (y2 - static_cast<float>(info.pad_y)) / info.scale;

          Detection detection;
          detection.x1 = std::clamp(x1, 0.0f, static_cast<float>(image_width - 1));
          detection.y1 = std::clamp(y1, 0.0f, static_cast<float>(image_height - 1));
          detection.x2 = std::clamp(x2, 0.0f, static_cast<float>(image_width - 1));
          detection.y2 = std::clamp(y2, 0.0f, static_cast<float>(image_height - 1));
          detection.confidence = confidence;
          detection.class_id = best_class;
          all_decoded.push_back(detection);
        }
      }
    }
  }

  return NonMaximumSuppressionPerClass(std::move(all_decoded), iou_threshold);
}

}  // namespace mtkdemo
