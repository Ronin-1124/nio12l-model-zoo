#pragma once

#include <array>
#include <string>
#include <vector>

namespace mtkdemo::yolo11n {

struct QuantParam {
  float scale = 1.0f;
  int zero_point = 0;
};

inline constexpr int kInputWidth = 1024;
inline constexpr int kInputHeight = 1024;
inline constexpr int kClassCount = 80;
inline constexpr int kRegMax = 16;
inline constexpr float kConfidenceThreshold = 0.25f;
inline constexpr float kIouThreshold = 0.45f;

inline const std::string kDefaultInt8ModelPath = "models/yolo11n/int8/yolo11n_int8.dla";
inline const std::string kDefaultFp32ModelPath = "models/yolo11n/fp32/yolo11n_fp32.dla";
inline constexpr QuantParam kInt8InputQuant = {0.00392157f, -128};

inline constexpr std::array<QuantParam, 3> kInt8BboxQuant = {
    QuantParam{0.0771554f, -50},
    QuantParam{0.0673739f, -52},
    QuantParam{0.0664506f, -40},
};

inline constexpr std::array<QuantParam, 3> kInt8ClsQuant = {
    QuantParam{0.0950655f, 127},
    QuantParam{0.11977f, 127},
    QuantParam{0.133898f, 119},
};

inline const std::vector<std::string> kDefaultImagePaths = {
    "assets/images/bus.jpg",
    "assets/images/zidane.jpg",
};

inline const std::string kDefaultOutputDir = "outputs/yolo11n";

}  // namespace mtkdemo::yolo11n
