#pragma once

#include <array>
#include <string>
#include <vector>

namespace mtkdemo::yolo11nobb {

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

inline const std::string kDefaultInt8ModelPath = "models/yolo11n-obb/int8/yolo11n-obb_int8.dla";
inline const std::string kDefaultFp32ModelPath = "models/yolo11n-obb/fp32/yolo11n-obb_fp32.dla";
inline constexpr QuantParam kInt8InputQuant = {0.00392157f, -128};

inline constexpr std::array<QuantParam, 3> kInt8AngleQuant = {
    QuantParam{0.0155005f, 33},
    QuantParam{0.00925453f, 5},
    QuantParam{0.0101462f, 29},
};

inline constexpr std::array<QuantParam, 3> kInt8ClsQuant = {
    QuantParam{0.211998f, 123},
    QuantParam{0.247605f, 121},
    QuantParam{0.166329f, 117},
};

inline constexpr std::array<QuantParam, 3> kInt8BboxQuant = {
    QuantParam{0.0658545f, -64},
    QuantParam{0.0604873f, -60},
    QuantParam{0.0595895f, -56},
};

inline const std::vector<std::string> kDefaultImagePaths = {
    "assets/images/boats.jpg",
};

inline const std::string kDefaultOutputDir = "outputs/yolo11n-obb";

}  // namespace mtkdemo::yolo11nobb
