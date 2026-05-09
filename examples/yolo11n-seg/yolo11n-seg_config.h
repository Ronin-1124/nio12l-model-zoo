#pragma once

#include <array>
#include <string>
#include <vector>

namespace mtkdemo::yolo11nseg {

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
inline constexpr int kRegMax = 16;
inline constexpr float kConfidenceThreshold = 0.25f;
inline constexpr float kIouThreshold = 0.45f;
inline constexpr float kMaskThreshold = 0.6f;

inline const std::string kDefaultInt8ModelPath = "examples/yolo11n-seg/model/int8/yolo11n-seg_int8.dla";
inline const std::string kDefaultFp32ModelPath = "examples/yolo11n-seg/model/fp32/yolo11n-seg_fp32.dla";
inline constexpr QuantParam kInt8InputQuant = {0.00392157f, -128};

inline constexpr std::array<QuantParam, 3> kInt8CoeffQuant = {
    QuantParam{0.0193071f, 33},
    QuantParam{0.0196476f, 37},
    QuantParam{0.0176456f, 31},
};

inline constexpr std::array<QuantParam, 3> kInt8ClsQuant = {
    QuantParam{0.0972605f, 127},
    QuantParam{0.118529f, 127},
    QuantParam{0.143741f, 113},
};

inline constexpr std::array<QuantParam, 3> kInt8BboxQuant = {
    QuantParam{0.0904757f, -68},
    QuantParam{0.0765874f, -58},
    QuantParam{0.0675753f, -48},
};

inline constexpr QuantParam kInt8ProtoQuant = {0.02512f, -117};

inline const std::vector<std::string> kDefaultImagePaths = {
    "assets/images/bus.jpg",
    "assets/images/zidane.jpg",
};

inline const std::string kDefaultOutputDir = "outputs/yolo11n-seg";

}  // namespace mtkdemo::yolo11nseg
