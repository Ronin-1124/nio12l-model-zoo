#pragma once

#include <array>
#include <string>
#include <vector>

namespace mtkdemo::yolo26n {

struct QuantParam {
  float scale = 1.0f;
  int zero_point = 0;
};

inline constexpr int kInputWidth = 512;
inline constexpr int kInputHeight = 512;
inline constexpr int kClassCount = 80;
inline constexpr float kConfidenceThreshold = 0.25f;
inline constexpr float kIouThreshold = 0.45f;

inline const std::string kDefaultInt8ModelPath = "examples/yolo26n/model/int8/yolo26n_int8.dla";
inline const std::string kDefaultFp32ModelPath = "examples/yolo26n/model/fp32/yolo26n_fp32.dla";
inline constexpr QuantParam kInt8InputQuant = {0.00392157f, -128};

inline constexpr std::array<QuantParam, 3> kInt8BboxQuant = {
    QuantParam{0.0321421f, -116},
    QuantParam{0.0557373f, -128},
    QuantParam{0.061752f, -128},
};

inline constexpr std::array<QuantParam, 3> kInt8ClsQuant = {
    QuantParam{0.112974f, 127},
    QuantParam{0.498957f, 127},
    QuantParam{0.455034f, 127},
};

inline const std::vector<std::string> kDefaultImagePaths = {
    "assets/images/bus.jpg",
    "assets/images/zidane.jpg",
};

inline const std::string kDefaultOutputDir = "outputs/yolo26n";

}  // namespace mtkdemo::yolo26n
