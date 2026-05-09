#pragma once

#include <array>
#include <string>
#include <vector>

namespace mtkdemo::yolo26npose {

struct QuantParam {
  float scale = 1.0f;
  int zero_point = 0;
};

inline constexpr int kInputWidth = 512;
inline constexpr int kInputHeight = 512;
inline constexpr int kKeypointCount = 17;
inline constexpr float kConfidenceThreshold = 0.35f;
inline constexpr float kIouThreshold = 0.45f;
inline constexpr float kKeypointThreshold = 0.5f;

inline const std::string kDefaultInt8ModelPath = "models/yolo26n-pose/int8/yolo26n-pose_int8.dla";
inline const std::string kDefaultFp32ModelPath = "models/yolo26n-pose/fp32/yolo26n-pose_fp32.dla";
inline constexpr QuantParam kInt8InputQuant = {0.00392157f, -128};

inline constexpr std::array<QuantParam, 3> kInt8KptQuant = {
    QuantParam{0.104392f, 43},
    QuantParam{0.0806424f, 3},
    QuantParam{0.075429f, 5},
};

inline constexpr std::array<QuantParam, 3> kInt8BboxQuant = {
    QuantParam{0.11405f, -126},
    QuantParam{0.071009f, -128},
    QuantParam{0.0465881f, -128},
};

inline constexpr std::array<QuantParam, 3> kInt8ScoreQuant = {
    QuantParam{0.118771f, 127},
    QuantParam{0.215979f, 127},
    QuantParam{0.321442f, 119},
};

inline const std::vector<std::string> kDefaultImagePaths = {
    "assets/images/bus.jpg",
    "assets/images/zidane.jpg",
};

inline const std::string kDefaultOutputDir = "outputs/yolo26n-pose";

}  // namespace mtkdemo::yolo26npose

