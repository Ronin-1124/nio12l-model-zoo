#pragma once

#include <array>
#include <string>
#include <vector>

namespace mtkdemo::yolo26nseg {

struct QuantParam {
  float scale = 1.0f;
  int zero_point = 0;
};

inline constexpr int kInputWidth = 512;
inline constexpr int kInputHeight = 512;
inline constexpr int kProtoWidth = 128;
inline constexpr int kProtoHeight = 128;
inline constexpr int kProtoChannels = 32;
inline constexpr int kClassCount = 80;
inline constexpr float kConfidenceThreshold = 0.25f;
inline constexpr float kIouThreshold = 0.45f;
inline constexpr float kMaskThreshold = 0.6f;

inline const std::string kDefaultInt8ModelPath = "examples/yolo26n-seg/model/int8/yolo26n-seg_int8.dla";
inline const std::string kDefaultFp32ModelPath = "examples/yolo26n-seg/model/fp32/yolo26n-seg_fp32.dla";
inline constexpr QuantParam kInt8InputQuant = {0.00392157f, -128};

inline constexpr std::array<QuantParam, 3> kInt8CoeffQuant = {
    QuantParam{0.0393522f, 11},
    QuantParam{0.0365385f, 19},
    QuantParam{0.0370186f, 53},
};

inline constexpr std::array<QuantParam, 3> kInt8BboxQuant = {
    QuantParam{0.0287128f, -114},
    QuantParam{0.0418423f, -126},
    QuantParam{0.0627117f, -128},
};

inline constexpr std::array<QuantParam, 3> kInt8ClsQuant = {
    QuantParam{0.107398f, 127},
    QuantParam{0.387165f, 127},
    QuantParam{0.302358f, 127},
};

inline constexpr QuantParam kInt8ProtoQuant = {0.0183494f, -113};

inline const std::vector<std::string> kDefaultImagePaths = {
    "assets/images/bus.jpg",
    "assets/images/zidane.jpg",
};

inline const std::string kDefaultOutputDir = "outputs/yolo26n-seg";

}  // namespace mtkdemo::yolo26nseg
