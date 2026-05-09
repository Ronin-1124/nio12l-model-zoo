#include "common/cpp/yolo12_postprocess.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace mtkdemo {
namespace {

float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

float IoU(const Detection& a, const Detection& b) {
  const float x1 = std::max(a.x1, b.x1);
  const float y1 = std::max(a.y1, b.y1);
  const float x2 = std::min(a.x2, b.x2);
  const float y2 = std::min(a.y2, b.y2);
  const float iw = std::max(0.0f, x2 - x1);
  const float ih = std::max(0.0f, y2 - y1);
  const float inter = iw * ih;
  const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  const float denom = area_a + area_b - inter;
  return denom > 0.0f ? inter / denom : 0.0f;
}

std::vector<Detection> Nms(std::vector<Detection> detections, float iou_threshold, size_t max_det) {
  std::sort(detections.begin(), detections.end(),
            [](const Detection& l, const Detection& r) { return l.confidence > r.confidence; });
  std::vector<Detection> kept;
  std::vector<bool> suppressed(detections.size(), false);
  for (size_t i = 0; i < detections.size(); ++i) {
    if (suppressed[i]) continue;
    kept.push_back(detections[i]);
    if (kept.size() >= max_det) break;
    for (size_t j = i + 1; j < detections.size(); ++j) {
      if (suppressed[j] || detections[i].class_id != detections[j].class_id) continue;
      if (IoU(detections[i], detections[j]) > iou_threshold) suppressed[j] = true;
    }
  }
  return kept;
}

struct Head {
  const uint8_t* data = nullptr;
  size_t channels = 0;
  size_t h = 0;
  size_t w = 0;
  bool int8 = false;
  float scale = 1.0f;
  int zero_point = 0;
};

float HeadValue(const Head& head, size_t c, size_t y, size_t x) {
  const size_t idx = (c * head.h + y) * head.w + x;
  if (!head.int8) return reinterpret_cast<const float*>(head.data)[idx];
  const int8_t q = reinterpret_cast<const int8_t*>(head.data)[idx];
  return (static_cast<float>(q) - static_cast<float>(head.zero_point)) * head.scale;
}

float DecodeDfl(const Head& bbox_head, size_t base_channel, size_t y, size_t x, int reg_max) {
  std::vector<float> logits(static_cast<size_t>(reg_max), 0.0f);
  float max_logit = -1e30f;
  for (int i = 0; i < reg_max; ++i) {
    const float v = HeadValue(bbox_head, base_channel + static_cast<size_t>(i), y, x);
    logits[static_cast<size_t>(i)] = v;
    if (v > max_logit) max_logit = v;
  }
  float sum = 0.0f;
  for (int i = 0; i < reg_max; ++i) {
    logits[static_cast<size_t>(i)] = std::exp(logits[static_cast<size_t>(i)] - max_logit);
    sum += logits[static_cast<size_t>(i)];
  }
  if (sum <= 0.0f) return 0.0f;
  float expected = 0.0f;
  for (int i = 0; i < reg_max; ++i) {
    expected += (logits[static_cast<size_t>(i)] / sum) * static_cast<float>(i);
  }
  return expected;
}

std::vector<Detection> DecodeImpl(const std::vector<Head>& bbox_heads,
                                  const std::vector<Head>& cls_heads, const LetterboxInfo& info,
                                  int image_width, int image_height, float confidence_threshold,
                                  float iou_threshold, int class_count, int reg_max,
                                  int input_size) {
  if (bbox_heads.size() != cls_heads.size() || bbox_heads.empty()) {
    throw std::runtime_error("YOLO12 decode: bbox/cls head count mismatch");
  }
  const size_t bbox_channels = static_cast<size_t>(4 * reg_max);
  std::vector<Detection> decoded;
  decoded.reserve(2500);

  for (size_t head_idx = 0; head_idx < bbox_heads.size(); ++head_idx) {
    const Head& b = bbox_heads[head_idx];
    const Head& c = cls_heads[head_idx];
    if (b.h != b.w || c.h != c.w || b.h != c.h) {
      throw std::runtime_error("YOLO12 decode: head spatial mismatch");
    }
    if (b.channels != bbox_channels || c.channels != static_cast<size_t>(class_count)) {
      throw std::runtime_error("YOLO12 decode: head channel mismatch");
    }
    const int stride = input_size / static_cast<int>(b.h);
    if (!(stride == 8 || stride == 16 || stride == 32)) {
      throw std::runtime_error("YOLO12 decode: unsupported stride");
    }

    for (size_t y = 0; y < b.h; ++y) {
      for (size_t x = 0; x < b.w; ++x) {
        int best_class = -1;
        float best_score = 0.0f;
        for (int cls = 0; cls < class_count; ++cls) {
          const float score = Sigmoid(HeadValue(c, static_cast<size_t>(cls), y, x));
          if (score > best_score) {
            best_score = score;
            best_class = cls;
          }
        }
        if (best_score < confidence_threshold) continue;

        const float l = DecodeDfl(b, 0, y, x, reg_max);
        const float t = DecodeDfl(b, static_cast<size_t>(reg_max), y, x, reg_max);
        const float r = DecodeDfl(b, static_cast<size_t>(reg_max * 2), y, x, reg_max);
        const float bb = DecodeDfl(b, static_cast<size_t>(reg_max * 3), y, x, reg_max);

        const float cx = (static_cast<float>(x) + 0.5f) * static_cast<float>(stride);
        const float cy = (static_cast<float>(y) + 0.5f) * static_cast<float>(stride);
        float x1 = cx - l * static_cast<float>(stride);
        float y1 = cy - t * static_cast<float>(stride);
        float x2 = cx + r * static_cast<float>(stride);
        float y2 = cy + bb * static_cast<float>(stride);

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
        if (det.x2 > det.x1 && det.y2 > det.y1) decoded.push_back(det);
      }
    }
  }
  return Nms(std::move(decoded), iou_threshold, 300);
}

std::vector<Head> BuildFp32Heads(const std::vector<std::vector<uint8_t>>& outputs,
                                 size_t expected_channels) {
  std::vector<Head> heads;
  heads.reserve(outputs.size());
  for (const auto& out : outputs) {
    if (out.size() % sizeof(float) != 0) throw std::runtime_error("YOLO12 decode: output is not fp32");
    const size_t count = out.size() / sizeof(float);
    if (count % expected_channels != 0) {
      throw std::runtime_error("YOLO12 decode: output count not divisible by channels");
    }
    const size_t hw = count / expected_channels;
    const size_t side = static_cast<size_t>(std::sqrt(static_cast<double>(hw)));
    if (side * side != hw) throw std::runtime_error("YOLO12 decode: output grid not square");
    heads.push_back(Head{out.data(), expected_channels, side, side, false, 1.0f, 0});
  }
  return heads;
}

std::vector<Head> BuildInt8Heads(const std::vector<std::vector<uint8_t>>& outputs,
                                 const std::vector<Yolo12QuantParam>& quant_params,
                                 size_t expected_channels) {
  if (outputs.size() != quant_params.size()) {
    throw std::runtime_error("YOLO12 int8 decode: outputs and quant mismatch");
  }
  std::vector<Head> heads;
  heads.reserve(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    const size_t count = outputs[i].size();
    if (count % expected_channels != 0) {
      throw std::runtime_error("YOLO12 int8 decode: output count not divisible by channels");
    }
    const size_t hw = count / expected_channels;
    const size_t side = static_cast<size_t>(std::sqrt(static_cast<double>(hw)));
    if (side * side != hw) throw std::runtime_error("YOLO12 int8 decode: output grid not square");
    heads.push_back(Head{outputs[i].data(), expected_channels, side, side, true, quant_params[i].scale,
                         quant_params[i].zero_point});
  }
  return heads;
}

}  // namespace

std::vector<Detection> DecodeYolo12MultiOutput(
    const std::vector<std::vector<uint8_t>>& bbox_outputs,
    const std::vector<std::vector<uint8_t>>& cls_outputs, const LetterboxInfo& info,
    int image_width, int image_height, float confidence_threshold, float iou_threshold,
    int class_count, int reg_max, int input_size) {
  const auto bbox_heads = BuildFp32Heads(bbox_outputs, static_cast<size_t>(4 * reg_max));
  const auto cls_heads = BuildFp32Heads(cls_outputs, static_cast<size_t>(class_count));
  return DecodeImpl(bbox_heads, cls_heads, info, image_width, image_height, confidence_threshold,
                    iou_threshold, class_count, reg_max, input_size);
}

std::vector<Detection> DecodeYolo12MultiOutputInt8(
    const std::vector<std::vector<uint8_t>>& bbox_outputs,
    const std::vector<Yolo12QuantParam>& bbox_quant_params,
    const std::vector<std::vector<uint8_t>>& cls_outputs,
    const std::vector<Yolo12QuantParam>& cls_quant_params, const LetterboxInfo& info,
    int image_width, int image_height, float confidence_threshold, float iou_threshold,
    int class_count, int reg_max, int input_size) {
  const auto bbox_heads =
      BuildInt8Heads(bbox_outputs, bbox_quant_params, static_cast<size_t>(4 * reg_max));
  const auto cls_heads =
      BuildInt8Heads(cls_outputs, cls_quant_params, static_cast<size_t>(class_count));
  return DecodeImpl(bbox_heads, cls_heads, info, image_width, image_height, confidence_threshold,
                    iou_threshold, class_count, reg_max, input_size);
}

}  // namespace mtkdemo
