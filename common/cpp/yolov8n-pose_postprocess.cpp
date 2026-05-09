  #include "common/cpp/yolov8n-pose_postprocess.h"

  #include <algorithm>
  #include <cmath>
  #include <stdexcept>
  #include <vector>

  namespace mtkdemo {
  namespace {

  struct Candidate {
    Detection det;
    std::vector<float> kpts;  // keypoint_count * 3
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
        if (suppressed[j]) continue;
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

  float DecodeDfl(const Head& head, size_t base_channel, size_t y, size_t x, int reg_max) {
    std::vector<float> logits(static_cast<size_t>(reg_max), 0.0f);
    float max_logit = -1e30f;
    for (int i = 0; i < reg_max; ++i) {
      const float v = HeadValue(head, base_channel + static_cast<size_t>(i), y, x);
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
      if (out.size() % sizeof(float) != 0) throw std::runtime_error("pose decode: non-fp32 tensor");
      const size_t count = out.size() / sizeof(float);
      if (count % channels != 0) throw std::runtime_error("pose decode: invalid channels");
      const size_t hw = count / channels;
      const size_t side = static_cast<size_t>(std::sqrt(static_cast<double>(hw)));
      if (side * side != hw) throw std::runtime_error("pose decode: non-square head");
      heads.push_back(Head{out.data(), channels, side, side, false, 1.0f, 0});
    }
    return heads;
  }

  std::vector<Head> BuildInt8Heads(const std::vector<std::vector<uint8_t>>& outputs,
                                  const std::vector<YoloV8PoseQuantParam>& q, size_t channels) {
    if (outputs.size() != q.size()) throw std::runtime_error("pose decode: quant size mismatch");
    std::vector<Head> heads;
    heads.reserve(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      const size_t count = outputs[i].size();
      if (count % channels != 0) throw std::runtime_error("pose decode: invalid channels");
      const size_t hw = count / channels;
      const size_t side = static_cast<size_t>(std::sqrt(static_cast<double>(hw)));
      if (side * side != hw) throw std::runtime_error("pose decode: non-square head");
      heads.push_back(Head{outputs[i].data(), channels, side, side, true, q[i].scale, q[i].zero_point});
    }
    return heads;
  }

  std::vector<PoseInstance> DecodeImpl(const std::vector<Head>& kpt_heads,
                                      const std::vector<Head>& score_heads,
                                      const std::vector<Head>& bbox_heads, const LetterboxInfo& info,
                                      int image_width, int image_height, float confidence_threshold,
                                      float iou_threshold, int keypoint_count, int reg_max,
                                      int input_size) {
    if (kpt_heads.size() != score_heads.size() || score_heads.size() != bbox_heads.size() ||
        kpt_heads.empty()) {
      throw std::runtime_error("pose decode: head count mismatch");
    }
    const size_t kpt_channels = static_cast<size_t>(keypoint_count * 3);
    const size_t bbox_channels = static_cast<size_t>(reg_max * 4);

    std::vector<Candidate> candidates;
    candidates.reserve(5000);

    for (size_t head_idx = 0; head_idx < bbox_heads.size(); ++head_idx) {
      const Head& k = kpt_heads[head_idx];
      const Head& s = score_heads[head_idx];
      const Head& b = bbox_heads[head_idx];
      if (k.channels != kpt_channels || s.channels != 1 || b.channels != bbox_channels || k.h != s.h ||
          k.w != s.w || b.h != s.h || b.w != s.w) {
        throw std::runtime_error("pose decode: head shape mismatch");
      }
      const int stride = input_size / static_cast<int>(b.h);
      if (!(stride == 8 || stride == 16 || stride == 32)) {
        throw std::runtime_error("pose decode: unsupported stride");
      }

      for (size_t y = 0; y < b.h; ++y) {
        for (size_t x = 0; x < b.w; ++x) {
          const float score = Sigmoid(HeadValue(s, 0, y, x));
          if (score < confidence_threshold) continue;

          const float l = DecodeDfl(b, 0 * static_cast<size_t>(reg_max), y, x, reg_max);
          const float t = DecodeDfl(b, 1 * static_cast<size_t>(reg_max), y, x, reg_max);
          const float r = DecodeDfl(b, 2 * static_cast<size_t>(reg_max), y, x, reg_max);
          const float bb = DecodeDfl(b, 3 * static_cast<size_t>(reg_max), y, x, reg_max);

          const float cx = (static_cast<float>(x) + 0.5f) * static_cast<float>(stride);
          const float cy = (static_cast<float>(y) + 0.5f) * static_cast<float>(stride);

          Detection det;
          det.x1 = (cx - l * static_cast<float>(stride) - static_cast<float>(info.pad_x)) / info.scale;
          det.y1 = (cy - t * static_cast<float>(stride) - static_cast<float>(info.pad_y)) / info.scale;
          det.x2 = (cx + r * static_cast<float>(stride) - static_cast<float>(info.pad_x)) / info.scale;
          det.y2 = (cy + bb * static_cast<float>(stride) - static_cast<float>(info.pad_y)) / info.scale;
          det.x1 = std::clamp(det.x1, 0.0f, static_cast<float>(image_width - 1));
          det.y1 = std::clamp(det.y1, 0.0f, static_cast<float>(image_height - 1));
          det.x2 = std::clamp(det.x2, 0.0f, static_cast<float>(image_width - 1));
          det.y2 = std::clamp(det.y2, 0.0f, static_cast<float>(image_height - 1));
          det.class_id = 0;
          det.confidence = score;
          if (!(det.x2 > det.x1 && det.y2 > det.y1)) continue;

          Candidate cand;
          cand.det = det;
          cand.kpts.resize(static_cast<size_t>(keypoint_count) * 3U, 0.0f);
          for (int ki = 0; ki < keypoint_count; ++ki) {
            const size_t base = static_cast<size_t>(ki) * 3U;
            const float kx = HeadValue(k, base + 0, y, x);
            const float ky = HeadValue(k, base + 1, y, x);
            const float kv = HeadValue(k, base + 2, y, x);

            // YOLOv8 pose raw head decoding:
            // kpt_xy = (raw_xy * 2 + grid_xy) * stride
            const float px = (kx * 2.0f + static_cast<float>(x)) * static_cast<float>(stride);
            const float py = (ky * 2.0f + static_cast<float>(y)) * static_cast<float>(stride);
            const float pv = Sigmoid(kv);

            cand.kpts[base + 0] = (px - static_cast<float>(info.pad_x)) / info.scale;
            cand.kpts[base + 1] = (py - static_cast<float>(info.pad_y)) / info.scale;
            cand.kpts[base + 2] = pv;
          }
          candidates.push_back(std::move(cand));
        }
      }
    }

    auto kept = Nms(std::move(candidates), iou_threshold, 300);
    std::vector<PoseInstance> results;
    results.reserve(kept.size());
    for (auto& c : kept) {
      PoseInstance p;
      p.detection = c.det;
      p.keypoints = std::move(c.kpts);
      results.push_back(std::move(p));
    }
    return results;
  }

  }  // namespace

  std::vector<PoseInstance> DecodeYoloV8PoseMultiOutput(
      const std::vector<std::vector<uint8_t>>& kpt_outputs,
      const std::vector<std::vector<uint8_t>>& score_outputs,
      const std::vector<std::vector<uint8_t>>& bbox_outputs, const LetterboxInfo& info,
      int image_width, int image_height, float confidence_threshold, float iou_threshold,
      int keypoint_count, int reg_max, int input_size) {
    const auto kpt_heads = BuildFp32Heads(kpt_outputs, static_cast<size_t>(keypoint_count * 3));
    const auto score_heads = BuildFp32Heads(score_outputs, 1);
    const auto bbox_heads = BuildFp32Heads(bbox_outputs, static_cast<size_t>(reg_max * 4));
    return DecodeImpl(kpt_heads, score_heads, bbox_heads, info, image_width, image_height,
                      confidence_threshold, iou_threshold, keypoint_count, reg_max, input_size);
  }

  std::vector<PoseInstance> DecodeYoloV8PoseMultiOutputInt8(
      const std::vector<std::vector<uint8_t>>& kpt_outputs,
      const std::vector<YoloV8PoseQuantParam>& kpt_quant,
      const std::vector<std::vector<uint8_t>>& score_outputs,
      const std::vector<YoloV8PoseQuantParam>& score_quant,
      const std::vector<std::vector<uint8_t>>& bbox_outputs,
      const std::vector<YoloV8PoseQuantParam>& bbox_quant, const LetterboxInfo& info,
      int image_width, int image_height, float confidence_threshold, float iou_threshold,
      int keypoint_count, int reg_max, int input_size) {
    const auto kpt_heads = BuildInt8Heads(kpt_outputs, kpt_quant, static_cast<size_t>(keypoint_count * 3));
    const auto score_heads = BuildInt8Heads(score_outputs, score_quant, 1);
    const auto bbox_heads = BuildInt8Heads(bbox_outputs, bbox_quant, static_cast<size_t>(reg_max * 4));
    return DecodeImpl(kpt_heads, score_heads, bbox_heads, info, image_width, image_height,
                      confidence_threshold, iou_threshold, keypoint_count, reg_max, input_size);
  }

  }  // namespace mtkdemo

