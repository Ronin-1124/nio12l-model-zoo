#include "common/cpp/yolov8n-seg_postprocess.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace mtkdemo {
namespace {

struct Candidate {
  Detection det;
  std::vector<float> coeff;
};

struct Head {
  const uint8_t* data = nullptr;
  size_t channels = 0;
  size_t h = 0;
  size_t w = 0;
  bool int8 = false;
  float scale = 1.0f;
  int zero_point = 0;
};

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

std::vector<Candidate> Nms(std::vector<Candidate> detections, float iou_threshold, size_t max_det) {
  std::sort(detections.begin(), detections.end(),
            [](const Candidate& l, const Candidate& r) {
              return l.det.confidence > r.det.confidence;
            });
  std::vector<Candidate> kept;
  std::vector<bool> suppressed(detections.size(), false);
  for (size_t i = 0; i < detections.size(); ++i) {
    if (suppressed[i]) continue;
    kept.push_back(std::move(detections[i]));
    if (kept.size() >= max_det) break;
    for (size_t j = i + 1; j < detections.size(); ++j) {
      if (suppressed[j] || detections[i].det.class_id != detections[j].det.class_id) continue;
      if (IoU(detections[i].det, detections[j].det) > iou_threshold) suppressed[j] = true;
    }
  }
  return kept;
}

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
    max_logit = std::max(max_logit, v);
  }
  float sum = 0.0f;
  for (int i = 0; i < reg_max; ++i) {
    logits[static_cast<size_t>(i)] = std::exp(logits[static_cast<size_t>(i)] - max_logit);
    sum += logits[static_cast<size_t>(i)];
  }
  if (sum <= 0.0f) return 0.0f;
  float value = 0.0f;
  for (int i = 0; i < reg_max; ++i) {
    value += (logits[static_cast<size_t>(i)] / sum) * static_cast<float>(i);
  }
  return value;
}

std::vector<Head> BuildFp32Heads(const std::vector<std::vector<uint8_t>>& outputs, size_t channels) {
  std::vector<Head> heads;
  heads.reserve(outputs.size());
  for (const auto& out : outputs) {
    if (out.size() % sizeof(float) != 0) throw std::runtime_error("seg decode: non-fp32 tensor");
    const size_t count = out.size() / sizeof(float);
    if (count % channels != 0) throw std::runtime_error("seg decode: invalid channels");
    const size_t hw = count / channels;
    const size_t side = static_cast<size_t>(std::sqrt(static_cast<double>(hw)));
    if (side * side != hw) throw std::runtime_error("seg decode: non-square head");
    heads.push_back(Head{out.data(), channels, side, side, false, 1.0f, 0});
  }
  return heads;
}

std::vector<Head> BuildInt8Heads(const std::vector<std::vector<uint8_t>>& outputs,
                                 const std::vector<YoloV8SegQuantParam>& q, size_t channels) {
  if (outputs.size() != q.size()) throw std::runtime_error("seg decode: quant size mismatch");
  std::vector<Head> heads;
  heads.reserve(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    const size_t count = outputs[i].size();
    if (count % channels != 0) throw std::runtime_error("seg decode: invalid channels");
    const size_t hw = count / channels;
    const size_t side = static_cast<size_t>(std::sqrt(static_cast<double>(hw)));
    if (side * side != hw) throw std::runtime_error("seg decode: non-square head");
    heads.push_back(Head{outputs[i].data(), channels, side, side, true, q[i].scale, q[i].zero_point});
  }
  return heads;
}

std::vector<float> BuildMaskLogits(const std::vector<float>& coeff, const Head& proto) {
  std::vector<float> logits(proto.h * proto.w, 0.0f);
  for (size_t y = 0; y < proto.h; ++y) {
    for (size_t x = 0; x < proto.w; ++x) {
      float v = 0.0f;
      for (size_t c = 0; c < coeff.size(); ++c) {
        v += coeff[c] * HeadValue(proto, c, y, x);
      }
      logits[y * proto.w + x] = v;
    }
  }
  return logits;
}

std::vector<uint8_t> BuildMaskToOriginal(const std::vector<float>& mask_logits, const Detection& det,
                                         const LetterboxInfo& info, int image_width, int image_height,
                                         int input_size, int proto_h, int proto_w,
                                         float mask_threshold) {
  std::vector<uint8_t> mask(static_cast<size_t>(image_width) * static_cast<size_t>(image_height), 0);
  const int x1 = std::max(0, static_cast<int>(std::floor(det.x1)));
  const int y1 = std::max(0, static_cast<int>(std::floor(det.y1)));
  const int x2 = std::min(image_width - 1, static_cast<int>(std::ceil(det.x2)));
  const int y2 = std::min(image_height - 1, static_cast<int>(std::ceil(det.y2)));
  const float scale_x = static_cast<float>(proto_w) / static_cast<float>(input_size);
  const float scale_y = static_cast<float>(proto_h) / static_cast<float>(input_size);

  for (int y = y1; y <= y2; ++y) {
    const float iy = static_cast<float>(y) * info.scale + static_cast<float>(info.pad_y);
    if (iy < 0.0f || iy >= static_cast<float>(input_size)) continue;
    const float pyf = iy * scale_y;
    const int py0 = std::clamp(static_cast<int>(std::floor(pyf)), 0, proto_h - 1);
    const int py1 = std::clamp(py0 + 1, 0, proto_h - 1);
    const float wy = std::clamp(pyf - static_cast<float>(py0), 0.0f, 1.0f);
    for (int x = x1; x <= x2; ++x) {
      const float ix = static_cast<float>(x) * info.scale + static_cast<float>(info.pad_x);
      if (ix < 0.0f || ix >= static_cast<float>(input_size)) continue;
      const float pxf = ix * scale_x;
      const int px0 = std::clamp(static_cast<int>(std::floor(pxf)), 0, proto_w - 1);
      const int px1 = std::clamp(px0 + 1, 0, proto_w - 1);
      const float wx = std::clamp(pxf - static_cast<float>(px0), 0.0f, 1.0f);

      const float v00 = mask_logits[static_cast<size_t>(py0) * static_cast<size_t>(proto_w) +
                                    static_cast<size_t>(px0)];
      const float v01 = mask_logits[static_cast<size_t>(py0) * static_cast<size_t>(proto_w) +
                                    static_cast<size_t>(px1)];
      const float v10 = mask_logits[static_cast<size_t>(py1) * static_cast<size_t>(proto_w) +
                                    static_cast<size_t>(px0)];
      const float v11 = mask_logits[static_cast<size_t>(py1) * static_cast<size_t>(proto_w) +
                                    static_cast<size_t>(px1)];
      const float top = v00 * (1.0f - wx) + v01 * wx;
      const float bottom = v10 * (1.0f - wx) + v11 * wx;
      const float prob = Sigmoid(top * (1.0f - wy) + bottom * wy);
      if (prob >= mask_threshold) {
        mask[static_cast<size_t>(y) * static_cast<size_t>(image_width) + static_cast<size_t>(x)] = 255;
      }
    }
  }
  return mask;
}

std::vector<SegmentationInstance> DecodeImpl(const std::vector<Head>& coeff_heads,
                                             const std::vector<Head>& cls_heads,
                                             const std::vector<Head>& bbox_heads, const Head& proto_head,
                                             const LetterboxInfo& info, int image_width, int image_height,
                                             float confidence_threshold, float iou_threshold,
                                             float mask_threshold, int class_count, int reg_max,
                                             int input_size, int proto_h, int proto_w) {
  if (coeff_heads.size() != 3 || cls_heads.size() != 3 || bbox_heads.size() != 3) {
    throw std::runtime_error("seg decode: expect 3-scale outputs");
  }
  if (proto_head.channels != coeff_heads[0].channels || static_cast<int>(proto_head.h) != proto_h ||
      static_cast<int>(proto_head.w) != proto_w) {
    throw std::runtime_error("seg decode: invalid proto shape");
  }

  std::vector<Candidate> candidates;
  candidates.reserve(4000);
  for (size_t i = 0; i < 3; ++i) {
    const Head& coeff = coeff_heads[i];
    const Head& cls = cls_heads[i];
    const Head& bbox = bbox_heads[i];
    if (coeff.h != cls.h || coeff.h != bbox.h || coeff.w != cls.w || coeff.w != bbox.w) {
      throw std::runtime_error("seg decode: head spatial mismatch");
    }
    if (coeff.channels != proto_head.channels || cls.channels != static_cast<size_t>(class_count) ||
        bbox.channels != static_cast<size_t>(4 * reg_max)) {
      throw std::runtime_error("seg decode: head channel mismatch");
    }
    const int stride = input_size / static_cast<int>(bbox.h);
    if (!(stride == 8 || stride == 16 || stride == 32)) throw std::runtime_error("seg decode: stride");

    for (size_t y = 0; y < bbox.h; ++y) {
      for (size_t x = 0; x < bbox.w; ++x) {
        int best_class = -1;
        float best_score = 0.0f;
        for (int c = 0; c < class_count; ++c) {
          const float s = Sigmoid(HeadValue(cls, static_cast<size_t>(c), y, x));
          if (s > best_score) {
            best_score = s;
            best_class = c;
          }
        }
        if (best_score < confidence_threshold) continue;

        const float l = DecodeDfl(bbox, 0, y, x, reg_max);
        const float t = DecodeDfl(bbox, static_cast<size_t>(reg_max), y, x, reg_max);
        const float r = DecodeDfl(bbox, static_cast<size_t>(2 * reg_max), y, x, reg_max);
        const float b = DecodeDfl(bbox, static_cast<size_t>(3 * reg_max), y, x, reg_max);
        const float cx = (static_cast<float>(x) + 0.5f) * static_cast<float>(stride);
        const float cy = (static_cast<float>(y) + 0.5f) * static_cast<float>(stride);

        Detection det;
        det.x1 = (cx - l * stride - static_cast<float>(info.pad_x)) / info.scale;
        det.y1 = (cy - t * stride - static_cast<float>(info.pad_y)) / info.scale;
        det.x2 = (cx + r * stride - static_cast<float>(info.pad_x)) / info.scale;
        det.y2 = (cy + b * stride - static_cast<float>(info.pad_y)) / info.scale;
        det.x1 = std::clamp(det.x1, 0.0f, static_cast<float>(image_width - 1));
        det.y1 = std::clamp(det.y1, 0.0f, static_cast<float>(image_height - 1));
        det.x2 = std::clamp(det.x2, 0.0f, static_cast<float>(image_width - 1));
        det.y2 = std::clamp(det.y2, 0.0f, static_cast<float>(image_height - 1));
        det.confidence = best_score;
        det.class_id = best_class;
        if (det.x2 <= det.x1 || det.y2 <= det.y1) continue;

        Candidate cand;
        cand.det = det;
        cand.coeff.resize(coeff.channels);
        for (size_t c = 0; c < coeff.channels; ++c) {
          cand.coeff[c] = HeadValue(coeff, c, y, x);
        }
        candidates.push_back(std::move(cand));
      }
    }
  }

  auto kept = Nms(std::move(candidates), iou_threshold, 300);
  std::vector<SegmentationInstance> results;
  results.reserve(kept.size());
  for (const auto& k : kept) {
    SegmentationInstance inst;
    inst.detection = k.det;
    const auto logits = BuildMaskLogits(k.coeff, proto_head);
    inst.mask = BuildMaskToOriginal(logits, inst.detection, info, image_width, image_height, input_size,
                                    proto_h, proto_w, mask_threshold);
    results.push_back(std::move(inst));
  }
  return results;
}

}  // namespace

std::vector<SegmentationInstance> DecodeYoloV8SegMultiOutput(
    const std::vector<std::vector<uint8_t>>& coeff_outputs,
    const std::vector<std::vector<uint8_t>>& cls_outputs,
    const std::vector<std::vector<uint8_t>>& bbox_outputs, const std::vector<uint8_t>& proto_output,
    const LetterboxInfo& info, int image_width, int image_height, float confidence_threshold,
    float iou_threshold, float mask_threshold, int class_count, int reg_max, int input_size,
    int proto_h, int proto_w) {
  auto coeff_heads = BuildFp32Heads(coeff_outputs, 32);
  auto cls_heads = BuildFp32Heads(cls_outputs, static_cast<size_t>(class_count));
  auto bbox_heads = BuildFp32Heads(bbox_outputs, static_cast<size_t>(4 * reg_max));
  const std::vector<std::vector<uint8_t>> proto_vec = {proto_output};
  auto proto_heads = BuildFp32Heads(proto_vec, 32);
  return DecodeImpl(coeff_heads, cls_heads, bbox_heads, proto_heads[0], info, image_width,
                    image_height, confidence_threshold, iou_threshold, mask_threshold, class_count,
                    reg_max, input_size, proto_h, proto_w);
}

std::vector<SegmentationInstance> DecodeYoloV8SegMultiOutputInt8(
    const std::vector<std::vector<uint8_t>>& coeff_outputs,
    const std::vector<YoloV8SegQuantParam>& coeff_quant,
    const std::vector<std::vector<uint8_t>>& cls_outputs,
    const std::vector<YoloV8SegQuantParam>& cls_quant,
    const std::vector<std::vector<uint8_t>>& bbox_outputs,
    const std::vector<YoloV8SegQuantParam>& bbox_quant, const std::vector<uint8_t>& proto_output,
    const YoloV8SegQuantParam& proto_quant, const LetterboxInfo& info, int image_width, int image_height,
    float confidence_threshold, float iou_threshold, float mask_threshold, int class_count, int reg_max,
    int input_size, int proto_h, int proto_w) {
  auto coeff_heads = BuildInt8Heads(coeff_outputs, coeff_quant, 32);
  auto cls_heads = BuildInt8Heads(cls_outputs, cls_quant, static_cast<size_t>(class_count));
  auto bbox_heads = BuildInt8Heads(bbox_outputs, bbox_quant, static_cast<size_t>(4 * reg_max));
  const std::vector<std::vector<uint8_t>> proto_vec = {proto_output};
  const std::vector<YoloV8SegQuantParam> proto_q = {proto_quant};
  auto proto_heads = BuildInt8Heads(proto_vec, proto_q, 32);
  return DecodeImpl(coeff_heads, cls_heads, bbox_heads, proto_heads[0], info, image_width, image_height,
                    confidence_threshold, iou_threshold, mask_threshold, class_count, reg_max,
                    input_size, proto_h, proto_w);
}

}  // namespace mtkdemo
