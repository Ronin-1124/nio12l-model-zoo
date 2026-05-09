#include "common/cpp/yolo26n-seg_postprocess.h"

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
      if (suppressed[j] || kept.back().det.class_id != detections[j].det.class_id) continue;
      if (IoU(kept.back().det, detections[j].det) > iou_threshold) suppressed[j] = true;
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

std::vector<Head> BuildFp32Heads(const std::vector<std::vector<uint8_t>>& outputs, size_t channels) {
  std::vector<Head> heads;
  heads.reserve(outputs.size());
  for (const auto& out : outputs) {
    if (out.size() % sizeof(float) != 0) throw std::runtime_error("seg26 decode: non-fp32 tensor");
    const size_t count = out.size() / sizeof(float);
    if (count % channels != 0) throw std::runtime_error("seg26 decode: invalid channels");
    const size_t hw = count / channels;
    const size_t side = static_cast<size_t>(std::sqrt(static_cast<double>(hw)));
    if (side * side != hw) throw std::runtime_error("seg26 decode: non-square head");
    heads.push_back(Head{out.data(), channels, side, side, false, 1.0f, 0});
  }
  return heads;
}

std::vector<Head> BuildInt8Heads(const std::vector<std::vector<uint8_t>>& outputs,
                                 const std::vector<YoloV8SegQuantParam>& q, size_t channels) {
  if (outputs.size() != q.size()) throw std::runtime_error("seg26 decode: quant size mismatch");
  std::vector<Head> heads;
  heads.reserve(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    const size_t count = outputs[i].size();
    if (count % channels != 0) throw std::runtime_error("seg26 decode: invalid channels");
    const size_t hw = count / channels;
    const size_t side = static_cast<size_t>(std::sqrt(static_cast<double>(hw)));
    if (side * side != hw) throw std::runtime_error("seg26 decode: non-square head");
    heads.push_back(Head{outputs[i].data(), channels, side, side, true, q[i].scale, q[i].zero_point});
  }
  return heads;
}

Head BuildProtoFp32(const std::vector<uint8_t>& proto, int proto_h, int proto_w, int channels) {
  if (proto.size() % sizeof(float) != 0) throw std::runtime_error("seg26 decode: non-fp32 proto");
  const size_t count = proto.size() / sizeof(float);
  const size_t expected = static_cast<size_t>(channels) * static_cast<size_t>(proto_h) *
                          static_cast<size_t>(proto_w);
  if (count != expected) throw std::runtime_error("seg26 decode: proto size mismatch");
  return Head{proto.data(), static_cast<size_t>(channels), static_cast<size_t>(proto_h),
              static_cast<size_t>(proto_w), false, 1.0f, 0};
}

Head BuildProtoInt8(const std::vector<uint8_t>& proto, int proto_h, int proto_w, int channels,
                    const YoloV8SegQuantParam& q) {
  const size_t expected = static_cast<size_t>(channels) * static_cast<size_t>(proto_h) *
                          static_cast<size_t>(proto_w);
  if (proto.size() != expected) throw std::runtime_error("seg26 decode: proto size mismatch");
  return Head{proto.data(), static_cast<size_t>(channels), static_cast<size_t>(proto_h),
              static_cast<size_t>(proto_w), true, q.scale, q.zero_point};
}

std::vector<float> BuildMaskLogits(const std::vector<float>& coeff, const Head& proto) {
  if (coeff.size() != proto.channels) throw std::runtime_error("seg26 decode: coeff/proto mismatch");
  const size_t hw = proto.h * proto.w;
  std::vector<float> out(hw, 0.0f);
  for (size_t c = 0; c < proto.channels; ++c) {
    for (size_t i = 0; i < hw; ++i) {
      const size_t y = i / proto.w;
      const size_t x = i % proto.w;
      out[i] += coeff[c] * HeadValue(proto, c, y, x);
    }
  }
  return out;
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
      const float v = top * (1.0f - wy) + bottom * wy;
      if (Sigmoid(v) >= mask_threshold) {
        mask[static_cast<size_t>(y) * static_cast<size_t>(image_width) + static_cast<size_t>(x)] = 255;
      }
    }
  }

  return mask;
}

std::vector<SegmentationInstance> DecodeImpl(const std::vector<Head>& coeff_heads,
                                             const std::vector<Head>& bbox_heads,
                                             const std::vector<Head>& cls_heads, const Head& proto,
                                             const LetterboxInfo& info, int image_width,
                                             int image_height, float confidence_threshold,
                                             float iou_threshold, float mask_threshold,
                                             int class_count, int input_size) {
  if (coeff_heads.size() != bbox_heads.size() || bbox_heads.size() != cls_heads.size() ||
      bbox_heads.empty()) {
    throw std::runtime_error("seg26 decode: head count mismatch");
  }

  std::vector<Candidate> candidates;
  candidates.reserve(6000);

  for (size_t head_idx = 0; head_idx < bbox_heads.size(); ++head_idx) {
    const Head& coef = coeff_heads[head_idx];
    const Head& b = bbox_heads[head_idx];
    const Head& c = cls_heads[head_idx];
    if (coef.channels != 32 || b.channels != 4 || c.channels != static_cast<size_t>(class_count) ||
        coef.h != b.h || coef.w != b.w || c.h != b.h || c.w != b.w) {
      throw std::runtime_error("seg26 decode: head shape mismatch");
    }
    const int stride = input_size / static_cast<int>(b.h);
    if (!(stride == 8 || stride == 16 || stride == 32)) {
      throw std::runtime_error("seg26 decode: unsupported stride");
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

        const float l = HeadValue(b, 0, y, x);
        const float t = HeadValue(b, 1, y, x);
        const float r = HeadValue(b, 2, y, x);
        const float bb = HeadValue(b, 3, y, x);

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
        if (!(det.x2 > det.x1 && det.y2 > det.y1)) continue;

        Candidate cand;
        cand.det = det;
        cand.coeff.resize(32);
        for (size_t cc = 0; cc < 32; ++cc) cand.coeff[cc] = HeadValue(coef, cc, y, x);
        candidates.push_back(std::move(cand));
      }
    }
  }

  auto kept = Nms(std::move(candidates), iou_threshold, 300);

  std::vector<SegmentationInstance> instances;
  instances.reserve(kept.size());
  for (auto& k : kept) {
    const std::vector<float> logits = BuildMaskLogits(k.coeff, proto);
    SegmentationInstance inst;
    inst.detection = k.det;
    inst.mask = BuildMaskToOriginal(logits, inst.detection, info, image_width, image_height, input_size,
                                    static_cast<int>(proto.h), static_cast<int>(proto.w),
                                    mask_threshold);
    instances.push_back(std::move(inst));
  }
  return instances;
}

}  // namespace

std::vector<SegmentationInstance> DecodeYolo26SegMultiOutput(
    const std::vector<std::vector<uint8_t>>& coeff_outputs,
    const std::vector<std::vector<uint8_t>>& bbox_outputs,
    const std::vector<std::vector<uint8_t>>& cls_outputs, const std::vector<uint8_t>& proto_output,
    const LetterboxInfo& info, int image_width, int image_height, float confidence_threshold,
    float iou_threshold, float mask_threshold, int class_count, int input_size, int proto_h,
    int proto_w) {
  const auto coeff_heads = BuildFp32Heads(coeff_outputs, 32);
  const auto bbox_heads = BuildFp32Heads(bbox_outputs, 4);
  const auto cls_heads = BuildFp32Heads(cls_outputs, static_cast<size_t>(class_count));
  const Head proto = BuildProtoFp32(proto_output, proto_h, proto_w, 32);
  return DecodeImpl(coeff_heads, bbox_heads, cls_heads, proto, info, image_width, image_height,
                    confidence_threshold, iou_threshold, mask_threshold, class_count, input_size);
}

std::vector<SegmentationInstance> DecodeYolo26SegMultiOutputInt8(
    const std::vector<std::vector<uint8_t>>& coeff_outputs,
    const std::vector<YoloV8SegQuantParam>& coeff_quant,
    const std::vector<std::vector<uint8_t>>& bbox_outputs,
    const std::vector<YoloV8SegQuantParam>& bbox_quant,
    const std::vector<std::vector<uint8_t>>& cls_outputs,
    const std::vector<YoloV8SegQuantParam>& cls_quant, const std::vector<uint8_t>& proto_output,
    const YoloV8SegQuantParam& proto_quant, const LetterboxInfo& info, int image_width,
    int image_height, float confidence_threshold, float iou_threshold, float mask_threshold,
    int class_count, int input_size, int proto_h, int proto_w) {
  const auto coeff_heads = BuildInt8Heads(coeff_outputs, coeff_quant, 32);
  const auto bbox_heads = BuildInt8Heads(bbox_outputs, bbox_quant, 4);
  const auto cls_heads = BuildInt8Heads(cls_outputs, cls_quant, static_cast<size_t>(class_count));
  const Head proto = BuildProtoInt8(proto_output, proto_h, proto_w, 32, proto_quant);
  return DecodeImpl(coeff_heads, bbox_heads, cls_heads, proto, info, image_width, image_height,
                    confidence_threshold, iou_threshold, mask_threshold, class_count, input_size);
}

}  // namespace mtkdemo

