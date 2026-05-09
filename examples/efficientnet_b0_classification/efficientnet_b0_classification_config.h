#pragma once

#include <string>
#include <vector>

#include "common/cpp/classification_demo.h"

namespace mtkdemo::efficientnetb0 {

inline constexpr int kInputWidth = 224;
inline constexpr int kInputHeight = 224;
inline constexpr int kInputChannels = 3;
// EfficientNet outputs 1000 pure ImageNet classes (no background), unlike TF-SLIM 1001-class MobileNet.
inline constexpr int kClassCount = 1000;
inline constexpr int kTopK = 5;
// Reuse the shared 1001-label table (idx 0 = background), so class i maps to labels[i + 1].
inline constexpr int kLabelOffset = 1;

inline const std::string kDefaultInt8ModelPath =
    "examples/efficientnet_b0_classification/model/int8/efficientnet_b0_int8.dla";
inline const std::string kDefaultFp32ModelPath =
    "examples/efficientnet_b0_classification/model/fp32/efficientnet_b0_fp32.dla";

// Matches `image_input` in `efficientnet_b0_mtk_int8.tflite`: real-valued range [0,1].
inline constexpr ClassificationQuantParam kInt8InputQuant = {0.00392157f, -128};
// Matches `probs`: scale=1/256, zp=-128, representing probabilities in [0,1].
inline constexpr ClassificationQuantParam kInt8OutputQuant = {0.00390625f, -128};

inline const std::vector<std::string> kDefaultImagePaths = {
    "assets/images/pickup.jpg",
    "assets/images/rhodesian_ridgeback.jpg",
};

inline const std::string kDefaultOutputDir = "outputs/efficientnet_b0_classification";

inline ClassificationDemoConfig Config() {
  return ClassificationDemoConfig{
      "efficientnet_b0_classification",
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
      ClassificationOutputKind::kProbabilities,
      ClassificationLabelMapping{kClassCount, kLabelOffset, /*output_tensor_size=*/0},
  };
}

}  // namespace mtkdemo::efficientnetb0
