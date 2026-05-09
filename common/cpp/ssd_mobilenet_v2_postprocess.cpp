#include "common/cpp/ssd_mobilenet_v2_postprocess.h"
#include "common/cpp/detection_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace mtkdemo {
namespace {

constexpr int kNumClasses = 91;
constexpr int kBoxValues = 4;

int8_t ByteToInt8(uint8_t b) { return static_cast<int8_t>(b); }

float DequantizeInt8(uint8_t byte, const SsdQuantParam& q) {
  if (q.scale <= 0.0f) {
    throw std::runtime_error("SSD int8 decode: quant scale must be positive");
  }
  const int32_t v = static_cast<int32_t>(ByteToInt8(byte));
  return (static_cast<float>(v) - static_cast<float>(q.zero_point)) * q.scale;
}

void DecodeAnchors(const float* boxes, const float* probs_base, size_t num_anchors, int image_width,
                   int image_height, float score_threshold, std::vector<Detection>* out) {
  for (size_t a = 0; a < num_anchors; ++a) {
    const float ymin = boxes[a * kBoxValues + 0];
    const float xmin = boxes[a * kBoxValues + 1];
    const float ymax = boxes[a * kBoxValues + 2];
    const float xmax = boxes[a * kBoxValues + 3];

    const float* probs = probs_base + a * static_cast<size_t>(kNumClasses);

    // raw_detection_scores already contains score-converted probabilities; skip TF SSD background class (id 0).
    int best_c = 0;
    float best_p = -1.0f;
    for (int c = 1; c < kNumClasses; ++c) {
      if (probs[static_cast<size_t>(c)] > best_p) {
        best_p = probs[static_cast<size_t>(c)];
        best_c = c;
      }
    }
    if (best_p < score_threshold) {
      continue;
    }

    const float ny1 = std::clamp(ymin, 0.0f, 1.0f);
    const float nx1 = std::clamp(xmin, 0.0f, 1.0f);
    const float ny2 = std::clamp(ymax, 0.0f, 1.0f);
    const float nx2 = std::clamp(xmax, 0.0f, 1.0f);

    Detection det;
    det.x1 = nx1 * static_cast<float>(image_width);
    det.y1 = ny1 * static_cast<float>(image_height);
    det.x2 = nx2 * static_cast<float>(image_width);
    det.y2 = ny2 * static_cast<float>(image_height);
    det.confidence = best_p;
    det.class_id = best_c;
    out->push_back(det);
  }
}

}  // namespace

std::vector<Detection> DecodeSsdMobilenetV2Raw(
    const std::vector<uint8_t>& raw_boxes_bytes, const std::vector<uint8_t>& raw_scores_bytes,
    int image_width, int image_height, float score_threshold, float iou_threshold, size_t top_k) {
  if (raw_boxes_bytes.size() % sizeof(float) != 0 || raw_scores_bytes.size() % sizeof(float) != 0) {
    throw std::runtime_error("SSD FP32 decode: buffers must be float32 aligned");
  }
  const size_t n_box_floats = raw_boxes_bytes.size() / sizeof(float);
  const size_t n_score_floats = raw_scores_bytes.size() / sizeof(float);
  if (n_box_floats % static_cast<size_t>(kBoxValues) != 0) {
    throw std::runtime_error("SSD FP32 decode: box element count not divisible by 4");
  }
  if (n_score_floats % static_cast<size_t>(kNumClasses) != 0) {
    throw std::runtime_error("SSD FP32 decode: score element count not divisible by 91");
  }
  const size_t num_anchors = n_box_floats / static_cast<size_t>(kBoxValues);
  if (num_anchors == 0 || num_anchors * static_cast<size_t>(kNumClasses) != n_score_floats) {
    throw std::runtime_error("SSD FP32 decode: box/score anchor count mismatch");
  }

  const float* boxes = reinterpret_cast<const float*>(raw_boxes_bytes.data());
  const float* scores = reinterpret_cast<const float*>(raw_scores_bytes.data());

  std::vector<Detection> raw_dets;
  raw_dets.reserve(num_anchors / 8);
  DecodeAnchors(boxes, scores, num_anchors, image_width, image_height, score_threshold, &raw_dets);

  std::vector<Detection> nms = NonMaximumSuppressionPerClass(std::move(raw_dets), iou_threshold);
  return LimitTopKDetections(std::move(nms), top_k);
}

std::vector<Detection> DecodeSsdMobilenetV2RawInt8(
    const std::vector<uint8_t>& raw_boxes_bytes, const std::vector<uint8_t>& raw_scores_bytes,
    const SsdQuantParam& boxes_quant, const SsdQuantParam& scores_quant, int image_width,
    int image_height, float score_threshold, float iou_threshold, size_t top_k) {
  if (raw_boxes_bytes.size() % static_cast<size_t>(kBoxValues) != 0) {
    throw std::runtime_error("SSD int8 decode: box byte count not divisible by 4");
  }
  if (raw_scores_bytes.size() % static_cast<size_t>(kNumClasses) != 0) {
    throw std::runtime_error("SSD int8 decode: score byte count not divisible by 91");
  }
  const size_t num_anchors = raw_boxes_bytes.size() / static_cast<size_t>(kBoxValues);
  if (num_anchors == 0 ||
      num_anchors * static_cast<size_t>(kNumClasses) != raw_scores_bytes.size()) {
    throw std::runtime_error("SSD int8 decode: box/score anchor count mismatch");
  }

  std::vector<float> boxes(num_anchors * static_cast<size_t>(kBoxValues));
  std::vector<float> scores(num_anchors * static_cast<size_t>(kNumClasses));

  for (size_t i = 0; i < raw_boxes_bytes.size(); ++i) {
    boxes[i] = DequantizeInt8(raw_boxes_bytes[i], boxes_quant);
  }
  for (size_t i = 0; i < raw_scores_bytes.size(); ++i) {
    scores[i] = DequantizeInt8(raw_scores_bytes[i], scores_quant);
  }

  std::vector<Detection> raw_dets;
  raw_dets.reserve(num_anchors / 8);
  DecodeAnchors(boxes.data(), scores.data(), num_anchors, image_width, image_height, score_threshold,
                &raw_dets);

  std::vector<Detection> nms = NonMaximumSuppressionPerClass(std::move(raw_dets), iou_threshold);
  return LimitTopKDetections(std::move(nms), top_k);
}

}  // namespace mtkdemo
