#pragma once

#include <array>
#include <string>
#include <vector>

namespace mtkdemo::yolov8npose {

struct QuantParam {
  float scale = 1.0f;
  int zero_point = 0;
};

inline constexpr int kInputWidth = 640;
inline constexpr int kInputHeight = 640;
inline constexpr int kKeypointCount = 17;
inline constexpr int kRegMax = 16;
inline constexpr float kConfidenceThreshold = 0.35f;
inline constexpr float kIouThreshold = 0.45f;
inline constexpr float kKeypointThreshold = 0.5f;

inline const std::string kDefaultInt8ModelPath = "models/yolov8n-pose/int8/yolov8n-pose_int8.dla";
inline const std::string kDefaultFp32ModelPath = "models/yolov8n-pose/fp32/yolov8n-pose_fp32.dla";
inline constexpr QuantParam kInt8InputQuant = {0.00392157f, -128};

inline constexpr std::array<QuantParam, 3> kInt8KptQuant = {
    QuantParam{0.0395558f, 11},
    QuantParam{0.0458481f, 15},
    QuantParam{0.0587877f, 11},
};

inline constexpr std::array<QuantParam, 3> kInt8ScoreQuant = {
    QuantParam{0.0845082f, 127},
    QuantParam{0.156336f, 121},
    QuantParam{0.150286f, 111},
};

inline constexpr std::array<QuantParam, 3> kInt8BboxQuant = {
    QuantParam{0.0530938f, -72},
    QuantParam{0.0574702f, -66},
    QuantParam{0.0632011f, -48},
};

inline const std::vector<std::string> kDefaultImagePaths = {
    "assets/images/bus.jpg",
    "assets/images/zidane.jpg",
};

inline const std::string kDefaultOutputDir = "outputs/yolov8n-pose";

}  // namespace mtkdemo::yolov8npose

