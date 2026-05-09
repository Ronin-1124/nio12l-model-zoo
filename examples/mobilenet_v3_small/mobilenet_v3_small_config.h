#pragma once

#include <string>
#include <vector>

#include "common/cpp/classification_demo.h"

namespace mtkdemo::mobilenetv3small {

inline constexpr int kInputWidth = 224;
inline constexpr int kInputHeight = 224;
inline constexpr int kInputChannels = 3;
inline constexpr int kClassCount = 1001;
inline constexpr int kTopK = 5;

inline const std::string kDefaultInt8ModelPath = "examples/mobilenet_v3_small/model/int8/mobilenet_v3_small_int8.dla";
inline const std::string kDefaultFp32ModelPath = "examples/mobilenet_v3_small/model/fp32/mobilenet_v3_small_fp32.dla";

inline constexpr ClassificationQuantParam kInt8InputQuant = {0.00392157f, -128};
inline constexpr ClassificationQuantParam kInt8OutputQuant = {0.0760708f, -62};

inline const std::vector<std::string> kDefaultImagePaths = {
    "assets/images/pickup.jpg",
    "assets/images/rhodesian_ridgeback.jpg",
};

inline const std::string kDefaultOutputDir = "outputs/mobilenet_v3_small";

inline ClassificationDemoConfig Config() {
  return ClassificationDemoConfig{
      "mobilenet_v3_small",
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

}  // namespace mtkdemo::mobilenetv3small
