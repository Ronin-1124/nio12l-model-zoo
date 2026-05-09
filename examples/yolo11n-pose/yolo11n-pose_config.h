#pragma once

#include <array>
#include <string>
#include <vector>

namespace mtkdemo::yolo11npose {

struct QuantParam {
  float scale = 1.0f;
  int zero_point = 0;
};

inline constexpr int kInputWidth = 512;
inline constexpr int kInputHeight = 512;
inline constexpr int kKeypointCount = 17;
inline constexpr int kRegMax = 16;
inline constexpr float kConfidenceThreshold = 0.35f;
inline constexpr float kIouThreshold = 0.45f;
inline constexpr float kKeypointThreshold = 0.5f;

inline const std::string kDefaultInt8ModelPath = "examples/yolo11n-pose/model/int8/yolo11n-pose_int8.dla";
inline const std::string kDefaultFp32ModelPath = "examples/yolo11n-pose/model/fp32/yolo11n-pose_fp32.dla";
inline constexpr QuantParam kInt8InputQuant = {0.00392157f, -128};

inline constexpr std::array<QuantParam, 3> kInt8KptQuant = {
    QuantParam{0.0430124f, -2},
    QuantParam{0.0455428f, 11},
    QuantParam{0.0614748f, 11},
};

inline constexpr std::array<QuantParam, 3> kInt8ScoreQuant = {
    QuantParam{0.0731007f, 127},
    QuantParam{0.11572f, 117},
    QuantParam{0.119232f, 105},
};

inline constexpr std::array<QuantParam, 3> kInt8BboxQuant = {
    QuantParam{0.0550104f, -72},
    QuantParam{0.0628285f, -56},
    QuantParam{0.0644776f, -54},
};

inline const std::vector<std::string> kDefaultImagePaths = {
    "assets/images/bus.jpg",
    "assets/images/zidane.jpg",
};

inline const std::string kDefaultOutputDir = "outputs/yolo11n-pose";

}  // namespace mtkdemo::yolo11npose

