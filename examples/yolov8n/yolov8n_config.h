 #pragma once

#include <array>
#include <string>
#include <vector>

namespace mtkdemo::yolov8n {

struct QuantParam {
  float scale = 1.0f;
  int zero_point = 0;
};

inline constexpr int kInputWidth = 640;
inline constexpr int kInputHeight = 640;
inline constexpr int kClassCount = 80;
inline constexpr int kRegMax = 16;
inline constexpr float kConfidenceThreshold = 0.25f;
inline constexpr float kIouThreshold = 0.45f;

inline const std::string kDefaultInt8ModelPath = "models/yolov8n/int8/yolov8n_int8.dla";
inline const std::string kDefaultFp32ModelPath = "models/yolov8n/fp32/yolov8n_fp32.dla";
inline constexpr QuantParam kInt8InputQuant = {0.00392157f, -128};
inline constexpr std::array<QuantParam, 3> kInt8OutputQuant = {
    QuantParam{0.159054f, 22},
    QuantParam{0.190915f, 58},
    QuantParam{0.199803f, 68},
};

inline const std::vector<std::string> kDefaultImagePaths = {
    "assets/images/bus.jpg",
    "assets/images/zidane.jpg",
};

inline const std::string kDefaultOutputDir = "outputs/yolov8n";

}  // namespace mtkdemo::yolov8n
