#pragma once

#include <array>
#include <string>
#include <vector>

namespace mtkdemo::yolov10n {

struct QuantParam {
  float scale = 1.0f;
  int zero_point = 0;
};

inline constexpr int kInputWidth = 512;
inline constexpr int kInputHeight = 512;
inline constexpr int kClassCount = 80;
inline constexpr int kRegMax = 16;
inline constexpr float kConfidenceThreshold = 0.25f;
inline constexpr float kIouThreshold = 0.45f;

inline const std::string kDefaultInt8ModelPath = "examples/yolov10n/model/int8/yolov10n_int8.dla";
inline const std::string kDefaultFp32ModelPath = "examples/yolov10n/model/fp32/yolov10n_fp32.dla";
inline constexpr QuantParam kInt8InputQuant = {0.00392157f, -128};

inline constexpr std::array<QuantParam, 3> kInt8BboxQuant = {
    QuantParam{0.0716183f, -68},
    QuantParam{0.0770334f, -76},
    QuantParam{0.0538869f, -64},
};

inline constexpr std::array<QuantParam, 3> kInt8ClsQuant = {
    QuantParam{0.102401f, 127},
    QuantParam{0.188377f, 127},
    QuantParam{0.176954f, 123},
};

inline const std::vector<std::string> kDefaultImagePaths = {
    "assets/images/bus.jpg",
    "assets/images/zidane.jpg",
};

inline const std::string kDefaultOutputDir = "outputs/yolov10n";

}  // namespace mtkdemo::yolov10n
