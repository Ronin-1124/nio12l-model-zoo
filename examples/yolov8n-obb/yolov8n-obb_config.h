#pragma once

#include <array>
#include <string>
#include <vector>

namespace mtkdemo::yolov8nobb {

struct QuantParam {
  float scale = 1.0f;
  int zero_point = 0;
};

inline constexpr int kInputWidth = 1024;
inline constexpr int kInputHeight = 1024;
inline constexpr int kClassCount = 15;
inline constexpr int kRegMax = 16;
inline constexpr float kConfidenceThreshold = 0.25f;
inline constexpr float kIouThreshold = 0.45f;

inline const std::string kDefaultInt8ModelPath = "models/yolov8n-obb/int8/yolov8n-obb_int8.dla";
inline const std::string kDefaultFp32ModelPath = "models/yolov8n-obb/fp32/yolov8n-obb_fp32.dla";
inline constexpr QuantParam kInt8InputQuant = {0.00392157f, -128};

inline constexpr std::array<QuantParam, 3> kInt8AngleQuant = {
    QuantParam{0.00925108f, 3},
    QuantParam{0.00889154f, 5},
    QuantParam{0.00965586f, 21},
};

inline constexpr std::array<QuantParam, 3> kInt8ClsQuant = {
    QuantParam{0.151341f, 127},
    QuantParam{0.128662f, 127},
    QuantParam{0.0970187f, 127},
};

inline constexpr std::array<QuantParam, 3> kInt8BboxQuant = {
    QuantParam{0.054203f, -66},
    QuantParam{0.0514644f, -72},
    QuantParam{0.0526552f, -52},
};

inline const std::vector<std::string> kDefaultImagePaths = {
    "assets/images/boats.jpg",
};

inline const std::string kDefaultOutputDir = "outputs/yolov8n-obb";

}  // namespace mtkdemo::yolov8nobb

