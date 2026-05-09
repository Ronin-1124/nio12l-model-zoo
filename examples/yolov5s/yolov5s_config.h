#pragma once

#include <array>
#include <string>
#include <vector>

namespace mtkdemo::yolov5s {

struct QuantParam {
  float scale = 1.0f;
  int zero_point = 0;
};

inline constexpr int kInputWidth = 640;
inline constexpr int kInputHeight = 640;
inline constexpr int kClassCount = 80;
inline constexpr float kConfidenceThreshold = 0.25f;
inline constexpr float kIouThreshold = 0.45f;

inline const std::string kDefaultInt8ModelPath = "examples/yolov5s/model/int8/yolov5s_int8.dla";
inline const std::string kDefaultFp32ModelPath = "examples/yolov5s/model/fp32/yolov5s_fp32.dla";
inline constexpr QuantParam kInt8InputQuant = {0.00392157f, -128};
inline constexpr std::array<QuantParam, 3> kInt8OutputQuant = {
    QuantParam{0.0885841f, 77},
    QuantParam{0.0780901f, 63},
    QuantParam{0.0868702f, 53},
};

inline const std::vector<std::string> kDefaultImagePaths = {
    "assets/images/bus.jpg",
    "assets/images/zidane.jpg",
};

inline const std::string kDefaultOutputDir = "outputs/yolov5s";

}  // namespace mtkdemo::yolov5s
