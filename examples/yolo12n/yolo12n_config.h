#pragma once

#include <array>
#include <string>
#include <vector>

namespace mtkdemo::yolo12n {

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

inline const std::string kDefaultInt8ModelPath = "examples/yolo12n/model/int8/yolo12n_int8.dla";
inline const std::string kDefaultFp32ModelPath = "examples/yolo12n/model/fp32/yolo12n_fp32.dla";
inline constexpr QuantParam kInt8InputQuant = {0.00392157f, -128};

inline constexpr std::array<QuantParam, 3> kInt8BboxQuant = {
    QuantParam{0.0759613f, -74},
    QuantParam{0.0619899f, -62},
    QuantParam{0.0619932f, -46},
};

inline constexpr std::array<QuantParam, 3> kInt8ClsQuant = {
    QuantParam{0.08817f, 127},
    QuantParam{0.121299f, 121},
    QuantParam{0.10786f, 107},
};

inline const std::vector<std::string> kDefaultImagePaths = {
    "assets/images/bus.jpg",
    "assets/images/zidane.jpg",
};

inline const std::string kDefaultOutputDir = "outputs/yolo12n";

}  // namespace mtkdemo::yolo12n
