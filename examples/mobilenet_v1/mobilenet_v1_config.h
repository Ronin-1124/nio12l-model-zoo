#pragma once

#include <string>
#include <vector>

#include "common/cpp/classification_demo.h"

namespace mtkdemo::mobilenetv1 {

struct QuantParam {
  float scale = 1.0f;
  int zero_point = 0;
};

inline constexpr int kInputWidth = 224;
inline constexpr int kInputHeight = 224;
inline constexpr int kInputChannels = 3;
inline constexpr int kClassCount = 1001;
inline constexpr int kTopK = 5;

inline const std::string kDefaultInt8ModelPath = "models/mobilenet_v1/int8/mobilenet_v1_int8.dla";
inline const std::string kDefaultFp32ModelPath = "models/mobilenet_v1/fp32/mobilenet_v1_fp32.dla";

inline constexpr QuantParam kInt8InputQuant = {0.00392157f, -128};
inline constexpr QuantParam kInt8OutputQuant = {0.139915f, -60};

inline const std::vector<std::string> kDefaultImagePaths = {
    "assets/images/pickup.jpg",
    "assets/images/rhodesian_ridgeback.jpg",
};

inline const std::string kDefaultOutputDir = "outputs/mobilenet_v1";

inline ClassificationDemoConfig Config() {
  return ClassificationDemoConfig{
      "mobilenet_v1",
      kDefaultInt8ModelPath,
      kDefaultFp32ModelPath,
      kDefaultImagePaths,
      kDefaultOutputDir,
      kInputWidth,
      kInputHeight,
      kInputChannels,
      kTopK,
      {kInt8InputQuant.scale, kInt8InputQuant.zero_point},
      {kInt8OutputQuant.scale, kInt8OutputQuant.zero_point},
      ClassificationOutputKind::kLogits,
      ClassificationLabelMapping{kClassCount, /*label_offset=*/0, /*output_tensor_size=*/0},
  };
}

}  // namespace mtkdemo::mobilenetv1
