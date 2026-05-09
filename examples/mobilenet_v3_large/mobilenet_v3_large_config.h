#pragma once

#include <string>
#include <vector>

#include "common/cpp/classification_demo.h"

namespace mtkdemo::mobilenetv3large {

inline constexpr int kInputWidth = 224;
inline constexpr int kInputHeight = 224;
inline constexpr int kInputChannels = 3;
inline constexpr int kClassCount = 1001;
inline constexpr int kTopK = 5;

inline const std::string kDefaultInt8ModelPath = "models/mobilenet_v3_large/int8/mobilenet_v3_large_int8.dla";
inline const std::string kDefaultFp32ModelPath = "models/mobilenet_v3_large/fp32/mobilenet_v3_large_fp32.dla";

inline constexpr ClassificationQuantParam kInt8InputQuant = {0.00392157f, -128};
inline constexpr ClassificationQuantParam kInt8OutputQuant = {0.063437f, -62};

inline const std::vector<std::string> kDefaultImagePaths = {
    "assets/images/pickup.jpg",
    "assets/images/rhodesian_ridgeback.jpg",
};

inline const std::string kDefaultOutputDir = "outputs/mobilenet_v3_large";

inline ClassificationDemoConfig Config() {
  return ClassificationDemoConfig{
      "mobilenet_v3_large",
      kDefaultInt8ModelPath,
      kDefaultFp32ModelPath,
      kDefaultImagePaths,
      kDefaultOutputDir,
      kInputWidth,
      kInputHeight,
      kInputChannels,
      kTopK,
      kInt8InputQuant,
      kInt8OutputQuant,
      ClassificationOutputKind::kLogits,
      ClassificationLabelMapping{kClassCount, /*label_offset=*/0, /*output_tensor_size=*/0},
  };
}

}  // namespace mtkdemo::mobilenetv3large
