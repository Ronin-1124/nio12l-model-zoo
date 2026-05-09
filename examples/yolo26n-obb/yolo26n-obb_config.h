#pragma once

#include <array>
#include <string>
#include <vector>

namespace mtkdemo::yolo26nobb {

struct QuantParam {
  float scale = 1.0f;
  int zero_point = 0;
};

inline constexpr int kInputWidth = 1024;
inline constexpr int kInputHeight = 1024;
inline constexpr int kClassCount = 15;
inline constexpr float kConfidenceThreshold = 0.25f;
inline constexpr float kIouThreshold = 0.45f;

inline const std::string kDefaultInt8ModelPath = "models/yolo26n-obb/int8/yolo26n-obb_int8.dla";
inline const std::string kDefaultFp32ModelPath = "models/yolo26n-obb/fp32/yolo26n-obb_fp32.dla";
inline constexpr QuantParam kInt8InputQuant = {0.00392157f, -128};

inline constexpr std::array<QuantParam, 3> kInt8AngleQuant = {
    QuantParam{0.00657899f, -14},
    QuantParam{0.00917035f, 1},
    QuantParam{0.00642374f, -10},
};

inline constexpr std::array<QuantParam, 3> kInt8BboxQuant = {
    QuantParam{0.025624f, -128},
    QuantParam{0.0412903f, -127},
    QuantParam{0.0750026f, -128},
};

inline constexpr std::array<QuantParam, 3> kInt8ClsQuant = {
    QuantParam{0.253907f, 127},
    QuantParam{1.12907f, 127},
    QuantParam{0.678608f, 127},
};

inline const std::vector<std::string> kDefaultImagePaths = {
    "assets/images/boats.jpg",
};

inline const std::string kDefaultOutputDir = "outputs/yolo26n-obb";

}  // namespace mtkdemo::yolo26nobb
