#pragma once

#include <array>
#include <string>
#include <vector>

namespace mtkdemo::yolov8nseg {

struct QuantParam {
  float scale = 1.0f;
  int zero_point = 0;
};

inline constexpr int kInputWidth = 640;
inline constexpr int kInputHeight = 640;
inline constexpr int kProtoWidth = 160;
inline constexpr int kProtoHeight = 160;
inline constexpr int kProtoChannels = 32;
inline constexpr int kClassCount = 80;
inline constexpr int kRegMax = 16;
inline constexpr float kConfidenceThreshold = 0.35f;
inline constexpr float kIouThreshold = 0.45f;
inline constexpr float kMaskThreshold = 0.6f;

inline const std::string kDefaultInt8ModelPath = "examples/yolov8n-seg/model/int8/yolov8n-seg_int8.dla";
inline const std::string kDefaultFp32ModelPath = "examples/yolov8n-seg/model/fp32/yolov8n-seg_fp32.dla";
inline constexpr QuantParam kInt8InputQuant = {0.00392157f, -128};

inline constexpr std::array<QuantParam, 3> kInt8CoeffQuant = {
    QuantParam{0.0190187f, 29},
    QuantParam{0.0171688f, 33},
    QuantParam{0.018701f, 51},
};

inline constexpr std::array<QuantParam, 3> kInt8ClsQuant = {
    QuantParam{0.0948619f, 127},
    QuantParam{0.121946f, 127},
    QuantParam{0.141998f, 117},
};

inline constexpr std::array<QuantParam, 3> kInt8BboxQuant = {
    QuantParam{0.0829791f, -72},
    QuantParam{0.0740273f, -50},
    QuantParam{0.0654985f, -48},
};

inline constexpr QuantParam kInt8ProtoQuant = {0.0227875f, -116};

inline const std::vector<std::string> kDefaultImagePaths = {
    "assets/images/bus.jpg",
    "assets/images/zidane.jpg",
};

inline const std::string kDefaultOutputDir = "outputs/yolov8n-seg";

}  // namespace mtkdemo::yolov8nseg
