#include "common/cpp/yolov8_postprocess.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace mtkdemo {
namespace {

constexpr int kYoloV8InputSize = 640;

float IntersectionOverUnion(const Detection& a, const Detection& b) {
  const float inter_x1 = std::max(a.x1, b.x1);
  const float inter_y1 = std::max(a.y1, b.y1);
  const float inter_x2 = std::min(a.x2, b.x2);
  const float inter_y2 = std::min(a.y2, b.y2);
  const float inter_w = std::max(0.0f, inter_x2 - inter_x1);
  const float inter_h = std::max(0.0f, inter_y2 - inter_y1);
  const float inter_area = inter_w * inter_h;
  const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  const float denom = area_a + area_b - inter_area;
  return denom > 0.0f ? inter_area / denom : 0.0f;
}

std::vector<Detection> NonMaximumSuppression(std::vector<Detection> detections,
                                              float iou_threshold, size_t max_detections) {
  std::sort(detections.begin(), detections.end(),
            [](const Detection& lhs, const Detection& rhs) {
              return lhs.confidence > rhs.confidence;
            });

  std::vector<Detection> kept;
  std::vector<bool> suppressed(detections.size(), false);
  for (size_t i = 0; i < detections.size(); ++i) {
    if (suppressed[i]) {
      continue;
    }
    kept.push_back(detections[i]);
    if (kept.size() >= max_detections) {
      break;
    }
    for (size_t j = i + 1; j < detections.size(); ++j) {
      if (suppressed[j] || detections[i].class_id != detections[j].class_id) {
        continue;
      }
      if (IntersectionOverUnion(detections[i], detections[j]) > iou_threshold) {
        suppressed[j] = true;
      }
    }
  }
  return kept;
}

struct OutputHead {
  const uint8_t* raw = nullptr;
  size_t channels = 0;
  size_t height = 0;
  size_t width = 0;
  bool int8 = false;
  float scale = 1.0f;
  int zero_point = 0;
};

float HeadValue(const OutputHead& head, size_t c, size_t h, size_t w) {
  const size_t idx = (c * head.height + h) * head.width + w;
  if (head.int8) {
    const int8_t q = reinterpret_cast<const int8_t*>(head.raw)[idx];
    return (static_cast<float>(q) - static_cast<float>(head.zero_point)) * head.scale;
  }
  return reinterpret_cast<const float*>(head.raw)[idx];
}

float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

float DecodeDfl(const OutputHead& head, size_t base_channel, size_t h, size_t w, int reg_max) {
  std::vector<float> logits(static_cast<size_t>(reg_max), 0.0f);
  float max_logit = -1e30f;
  for (int i = 0; i < reg_max; ++i) {
    const float v = HeadValue(head, base_channel + static_cast<size_t>(i), h, w);
    logits[static_cast<size_t>(i)] = v;
    if (v > max_logit) {
      max_logit = v;
    }
  }
  float sum = 0.0f;
  for (int i = 0; i < reg_max; ++i) {
    logits[static_cast<size_t>(i)] = std::exp(logits[static_cast<size_t>(i)] - max_logit);
    sum += logits[static_cast<size_t>(i)];
  }
  if (sum <= 0.0f) {
    return 0.0f;
  }
  float expected = 0.0f;
  for (int i = 0; i < reg_max; ++i) {
    expected += (logits[static_cast<size_t>(i)] / sum) * static_cast<float>(i);
  }
  return expected;
}

std::vector<Detection> DecodeYoloV8FromHeads(const std::vector<OutputHead>& heads,
                                             const LetterboxInfo& info, int image_width,
                                             int image_height, float confidence_threshold,
                                             float iou_threshold, int class_count, int reg_max) {
  if (reg_max <= 1 || class_count <= 0) {
    throw std::runtime_error("YOLOv8 decode: invalid reg_max/class_count");
  }
  const size_t reg_channels = static_cast<size_t>(reg_max * 4);
  const size_t expected_channels = reg_channels + static_cast<size_t>(class_count);
  std::vector<Detection> decoded;
  decoded.reserve(3000);

  for (const auto& head : heads) {
    if (head.channels != expected_channels || head.height != head.width) {
      throw std::runtime_error("YOLOv8 decode: unexpected output head shape");
    }
    const int stride = kYoloV8InputSize / static_cast<int>(head.height);
    if (!(stride == 8 || stride == 16 || stride == 32)) {
      throw std::runtime_error("YOLOv8 decode: unsupported stride");
    }

    for (size_t y = 0; y < head.height; ++y) {
      for (size_t x = 0; x < head.width; ++x) {
        int best_class = -1;
        float best_score = 0.0f;
        for (int c = 0; c < class_count; ++c) {
          const float score = Sigmoid(HeadValue(head, reg_channels + static_cast<size_t>(c), y, x));
          if (score > best_score) {
            best_score = score;
            best_class = c;
          }
        }
        if (best_score < confidence_threshold) {
          continue;
        }

        const float l = DecodeDfl(head, 0, y, x, reg_max);
        const float t = DecodeDfl(head, static_cast<size_t>(reg_max), y, x, reg_max);
        const float r = DecodeDfl(head, static_cast<size_t>(reg_max * 2), y, x, reg_max);
        const float b = DecodeDfl(head, static_cast<size_t>(reg_max * 3), y, x, reg_max);

        const float cx = (static_cast<float>(x) + 0.5f) * static_cast<float>(stride);
        const float cy = (static_cast<float>(y) + 0.5f) * static_cast<float>(stride);
        float x1 = cx - l * static_cast<float>(stride);
        float y1 = cy - t * static_cast<float>(stride);
        float x2 = cx + r * static_cast<float>(stride);
        float y2 = cy + b * static_cast<float>(stride);

        x1 = (x1 - static_cast<float>(info.pad_x)) / info.scale;
        y1 = (y1 - static_cast<float>(info.pad_y)) / info.scale;
        x2 = (x2 - static_cast<float>(info.pad_x)) / info.scale;
        y2 = (y2 - static_cast<float>(info.pad_y)) / info.scale;

        Detection det;
        det.x1 = std::clamp(x1, 0.0f, static_cast<float>(image_width - 1));
        det.y1 = std::clamp(y1, 0.0f, static_cast<float>(image_height - 1));
        det.x2 = std::clamp(x2, 0.0f, static_cast<float>(image_width - 1));
        det.y2 = std::clamp(y2, 0.0f, static_cast<float>(image_height - 1));
        det.class_id = best_class;
        det.confidence = best_score;
        if (det.x2 > det.x1 && det.y2 > det.y1) {
          decoded.push_back(det);
        }
      }
    }
  }

  return NonMaximumSuppression(std::move(decoded), iou_threshold, 300);
}

}  // namespace

std::vector<Detection> DecodeYoloV8MultiOutput(const std::vector<std::vector<uint8_t>>& outputs,
                                               const LetterboxInfo& info, int image_width,
                                               int image_height, float confidence_threshold,
                                               float iou_threshold, int class_count,
                                               int reg_max) {
  if (outputs.empty()) {
    throw std::runtime_error("YOLOv8 decode: empty outputs");
  }
  std::vector<OutputHead> heads;
  heads.reserve(outputs.size());
  const size_t expected_channels = static_cast<size_t>(reg_max * 4 + class_count);
  for (const auto& out : outputs) {
    if (out.size() % sizeof(float) != 0) {
      throw std::runtime_error("YOLOv8 decode: output buffer is not float32");
    }
    const size_t value_count = out.size() / sizeof(float);
    if (value_count % expected_channels != 0) {
      throw std::runtime_error("YOLOv8 decode: output elements not divisible by channels");
    }
    const size_t hw = value_count / expected_channels;
    const size_t side = static_cast<size_t>(std::sqrt(static_cast<double>(hw)));
    if (side * side != hw) {
      throw std::runtime_error("YOLOv8 decode: output grid is not square");
    }
    heads.push_back(OutputHead{out.data(), expected_channels, side, side, false, 1.0f, 0});
  }
  return DecodeYoloV8FromHeads(heads, info, image_width, image_height, confidence_threshold,
                               iou_threshold, class_count, reg_max);
}

std::vector<Detection> DecodeYoloV8MultiOutputInt8(
    const std::vector<std::vector<uint8_t>>& outputs,
    const std::vector<YoloV8QuantParam>& quant_params, const LetterboxInfo& info, int image_width,
    int image_height, float confidence_threshold, float iou_threshold, int class_count,
    int reg_max) {
  if (outputs.empty() || outputs.size() != quant_params.size()) {
    throw std::runtime_error("YOLOv8 int8 decode: outputs and quant_params mismatch");
  }
  std::vector<OutputHead> heads;
  heads.reserve(outputs.size());
  const size_t expected_channels = static_cast<size_t>(reg_max * 4 + class_count);
  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto& out = outputs[i];
    const auto& q = quant_params[i];
    if (q.scale <= 0.0f) {
      throw std::runtime_error("YOLOv8 int8 decode: quant scale must be positive");
    }
    const size_t value_count = out.size();
    if (value_count % expected_channels != 0) {
      throw std::runtime_error("YOLOv8 int8 decode: output elements not divisible by channels");
    }
    const size_t hw = value_count / expected_channels;
    const size_t side = static_cast<size_t>(std::sqrt(static_cast<double>(hw)));
    if (side * side != hw) {
      throw std::runtime_error("YOLOv8 int8 decode: output grid is not square");
    }
    heads.push_back(OutputHead{out.data(), expected_channels, side, side, true, q.scale,
                               q.zero_point});
  }
  return DecodeYoloV8FromHeads(heads, info, image_width, image_height, confidence_threshold,
                               iou_threshold, class_count, reg_max);
}

}  // namespace mtkdemo
