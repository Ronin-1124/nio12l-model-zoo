#include "common/cpp/yolo26n-obb_postprocess.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace mtkdemo {
namespace {
constexpr float kPi = 3.14159265358979323846f;

struct Point {
  float x = 0.0f;
  float y = 0.0f;
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
    if (out.size() % sizeof(float) != 0) throw std::runtime_error("YOLO26n-OBB decode: non-fp32 tensor");
    const size_t count = out.size() / sizeof(float);
    if (count % channels != 0) throw std::runtime_error("YOLO26n-OBB decode: invalid channels");
    const size_t hw = count / channels;
    const size_t side = static_cast<size_t>(std::sqrt(static_cast<double>(hw)));
    if (side * side != hw) throw std::runtime_error("YOLO26n-OBB decode: non-square head");
    heads.push_back(Head{out.data(), channels, side, side, false, 1.0f, 0});
  }
  return heads;
}

std::vector<Head> BuildInt8Heads(const std::vector<std::vector<uint8_t>>& outputs,
                                 const std::vector<YoloObbQuantParam>& q, size_t channels) {
  if (outputs.size() != q.size()) throw std::runtime_error("YOLO26n-OBB decode: quant size mismatch");
  std::vector<Head> heads;
  heads.reserve(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    const size_t count = outputs[i].size();
    if (count % channels != 0) throw std::runtime_error("YOLO26n-OBB decode: invalid channels");
    const size_t hw = count / channels;
    const size_t side = static_cast<size_t>(std::sqrt(static_cast<double>(hw)));
    if (side * side != hw) throw std::runtime_error("YOLO26n-OBB decode: non-square head");
    heads.push_back(Head{outputs[i].data(), channels, side, side, true, q[i].scale, q[i].zero_point});
  }
  return heads;
}

std::vector<Point> ObbToCorners(const OrientedDetection& d) {
  const float c = std::cos(d.angle);
  const float s = std::sin(d.angle);
  const float hw = d.w * 0.5f;
  const float hh = d.h * 0.5f;
  std::vector<Point> pts(4);
  const float xs[4] = {-hw, hw, hw, -hw};
  const float ys[4] = {-hh, -hh, hh, hh};
  for (int i = 0; i < 4; ++i) {
    pts[static_cast<size_t>(i)] = Point{d.cx + xs[i] * c - ys[i] * s, d.cy + xs[i] * s + ys[i] * c};
  }
  return pts;
}

float PolygonArea(const std::vector<Point>& poly) {
  if (poly.size() < 3) return 0.0f;
  float area = 0.0f;
  for (size_t i = 0; i < poly.size(); ++i) {
    const Point& a = poly[i];
    const Point& b = poly[(i + 1) % poly.size()];
    area += a.x * b.y - b.x * a.y;
  }
  return std::abs(area) * 0.5f;
}

bool IsInside(const Point& p, const Point& a, const Point& b) {
  return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x) >= 0.0f;
}

Point IntersectSeg(const Point& p1, const Point& p2, const Point& q1, const Point& q2) {
  const float A1 = p2.y - p1.y;
  const float B1 = p1.x - p2.x;
  const float C1 = A1 * p1.x + B1 * p1.y;
  const float A2 = q2.y - q1.y;
  const float B2 = q1.x - q2.x;
  const float C2 = A2 * q1.x + B2 * q1.y;
  const float det = A1 * B2 - A2 * B1;
  if (std::abs(det) < 1e-8f) return p2;
  return Point{(B2 * C1 - B1 * C2) / det, (A1 * C2 - A2 * C1) / det};
}

std::vector<Point> PolygonClip(const std::vector<Point>& subject, const std::vector<Point>& clipper) {
  std::vector<Point> output = subject;
  for (size_t i = 0; i < clipper.size(); ++i) {
    const Point A = clipper[i];
    const Point B = clipper[(i + 1) % clipper.size()];
    std::vector<Point> input = output;
    output.clear();
    if (input.empty()) break;
    Point S = input.back();
    for (const Point& E : input) {
      const bool Ein = IsInside(E, A, B);
      const bool Sin = IsInside(S, A, B);
      if (Ein) {
        if (!Sin) output.push_back(IntersectSeg(S, E, A, B));
        output.push_back(E);
      } else if (Sin) {
        output.push_back(IntersectSeg(S, E, A, B));
      }
      S = E;
    }
  }
  return output;
}

float RotatedIoU(const OrientedDetection& a, const OrientedDetection& b) {
  const auto pa = ObbToCorners(a);
  const auto pb = ObbToCorners(b);
  const float area_a = PolygonArea(pa);
  const float area_b = PolygonArea(pb);
  if (area_a <= 0.0f || area_b <= 0.0f) return 0.0f;
  const auto inter_poly = PolygonClip(pa, pb);
  const float inter = PolygonArea(inter_poly);
  const float denom = area_a + area_b - inter;
  return denom > 0.0f ? inter / denom : 0.0f;
}

std::vector<OrientedDetection> RotatedNms(std::vector<OrientedDetection> detections,
                                          float iou_threshold, size_t max_det) {
  std::sort(detections.begin(), detections.end(),
            [](const OrientedDetection& l, const OrientedDetection& r) {
              return l.confidence > r.confidence;
            });
  std::vector<OrientedDetection> kept;
  std::vector<bool> suppressed(detections.size(), false);
  for (size_t i = 0; i < detections.size(); ++i) {
    if (suppressed[i]) continue;
    kept.push_back(detections[i]);
    if (kept.size() >= max_det) break;
    for (size_t j = i + 1; j < detections.size(); ++j) {
      if (suppressed[j] || detections[j].class_id != detections[i].class_id) continue;
      if (RotatedIoU(detections[i], detections[j]) > iou_threshold) suppressed[j] = true;
    }
  }
  return kept;
}

std::vector<OrientedDetection> DecodeImpl(const std::vector<Head>& angle_heads,
                                          const std::vector<Head>& bbox_heads,
                                          const std::vector<Head>& cls_heads, const LetterboxInfo& info,
                                          int image_width, int image_height, float confidence_threshold,
                                          float iou_threshold, int class_count, int input_size) {
  if (angle_heads.size() != cls_heads.size() || cls_heads.size() != bbox_heads.size() ||
      angle_heads.empty()) {
    throw std::runtime_error("YOLO26n-OBB decode: head count mismatch");
  }
  std::vector<OrientedDetection> decoded;
  decoded.reserve(6000);
  for (size_t head_idx = 0; head_idx < bbox_heads.size(); ++head_idx) {
    const Head& a = angle_heads[head_idx];
    const Head& b = bbox_heads[head_idx];
    const Head& c = cls_heads[head_idx];
    if (a.channels != 1 || b.channels != 4 || c.channels != static_cast<size_t>(class_count) ||
        a.h != b.h || b.h != c.h || a.w != b.w || b.w != c.w) {
      throw std::runtime_error("YOLO26n-OBB decode: head shape mismatch");
    }
    const int stride = input_size / static_cast<int>(b.h);
    if (!(stride == 8 || stride == 16 || stride == 32)) {
      throw std::runtime_error("YOLO26n-OBB decode: unsupported stride");
    }
    for (size_t y = 0; y < b.h; ++y) {
      for (size_t x = 0; x < b.w; ++x) {
        int best_cls = -1;
        float best_score = 0.0f;
        for (int cls = 0; cls < class_count; ++cls) {
          const float s = Sigmoid(HeadValue(c, static_cast<size_t>(cls), y, x));
          if (s > best_score) {
            best_score = s;
            best_cls = cls;
          }
        }
        if (best_score < confidence_threshold) continue;

        const float l = HeadValue(b, 0, y, x);
        const float t = HeadValue(b, 1, y, x);
        const float r = HeadValue(b, 2, y, x);
        const float bb = HeadValue(b, 3, y, x);
        const float cx_in = (static_cast<float>(x) + 0.5f) * static_cast<float>(stride);
        const float cy_in = (static_cast<float>(y) + 0.5f) * static_cast<float>(stride);
        const float raw_angle = HeadValue(a, 0, y, x);
        const float theta = raw_angle;
        const float c_theta = std::cos(theta);
        const float s_theta = std::sin(theta);
        const float dx = (r - l) * 0.5f * static_cast<float>(stride);
        const float dy = (bb - t) * 0.5f * static_cast<float>(stride);
        const float det_cx_in = cx_in + dx * c_theta - dy * s_theta;
        const float det_cy_in = cy_in + dx * s_theta + dy * c_theta;

        OrientedDetection det;
        det.cx = (det_cx_in - static_cast<float>(info.pad_x)) / info.scale;
        det.cy = (det_cy_in - static_cast<float>(info.pad_y)) / info.scale;
        det.w = ((l + r) * static_cast<float>(stride)) / info.scale;
        det.h = ((t + bb) * static_cast<float>(stride)) / info.scale;
        det.cx = std::clamp(det.cx, 0.0f, static_cast<float>(image_width - 1));
        det.cy = std::clamp(det.cy, 0.0f, static_cast<float>(image_height - 1));
        det.w = std::clamp(det.w, 1.0f, static_cast<float>(image_width));
        det.h = std::clamp(det.h, 1.0f, static_cast<float>(image_height));
        det.angle = theta;
        det.class_id = best_cls;
        det.confidence = best_score;
        decoded.push_back(det);
      }
    }
  }
  return RotatedNms(std::move(decoded), iou_threshold, 300);
}

}  // namespace

std::vector<OrientedDetection> DecodeYolo26ObbMultiOutput(
    const std::vector<std::vector<uint8_t>>& angle_outputs,
    const std::vector<std::vector<uint8_t>>& bbox_outputs,
    const std::vector<std::vector<uint8_t>>& cls_outputs, const LetterboxInfo& info,
    int image_width, int image_height, float confidence_threshold, float iou_threshold,
    int class_count, int input_size) {
  const auto angle_heads = BuildFp32Heads(angle_outputs, 1);
  const auto bbox_heads = BuildFp32Heads(bbox_outputs, 4);
  const auto cls_heads = BuildFp32Heads(cls_outputs, static_cast<size_t>(class_count));
  return DecodeImpl(angle_heads, bbox_heads, cls_heads, info, image_width, image_height,
                    confidence_threshold, iou_threshold, class_count, input_size);
}

std::vector<OrientedDetection> DecodeYolo26ObbMultiOutputInt8(
    const std::vector<std::vector<uint8_t>>& angle_outputs,
    const std::vector<YoloObbQuantParam>& angle_quant,
    const std::vector<std::vector<uint8_t>>& bbox_outputs,
    const std::vector<YoloObbQuantParam>& bbox_quant,
    const std::vector<std::vector<uint8_t>>& cls_outputs,
    const std::vector<YoloObbQuantParam>& cls_quant, const LetterboxInfo& info, int image_width,
    int image_height, float confidence_threshold, float iou_threshold, int class_count,
    int input_size) {
  const auto angle_heads = BuildInt8Heads(angle_outputs, angle_quant, 1);
  const auto bbox_heads = BuildInt8Heads(bbox_outputs, bbox_quant, 4);
  const auto cls_heads = BuildInt8Heads(cls_outputs, cls_quant, static_cast<size_t>(class_count));
  return DecodeImpl(angle_heads, bbox_heads, cls_heads, info, image_width, image_height,
                    confidence_threshold, iou_threshold, class_count, input_size);
}

}  // namespace mtkdemo
