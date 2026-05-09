#pragma once

#include <string>
#include <vector>

#include "common/cpp/classification_demo.h"

namespace mtkdemo::resnet50 {

inline constexpr int kInputWidth = 224;
inline constexpr int kInputHeight = 224;
inline constexpr int kInputChannels = 3;
// The model uses 0-based ImageNet 1000-class indices (e.g. 717 = pickup), while the output tensor has length 1001,
// where the extra last element is a dummy placeholder. Effective class count = 1000, output tensor size = 1001.
inline constexpr int kClassCount = 1000;
inline constexpr int kOutputTensorSize = 1001;
inline constexpr int kTopK = 5;
// Reuse the shared 1001-label table (idx 0 = background): map model class id i to labels[i + 1].
inline constexpr int kLabelOffset = 1;

inline const std::string kDefaultInt8ModelPath = "examples/resnet50/model/int8/resnet50_int8.dla";
inline const std::string kDefaultFp32ModelPath = "examples/resnet50/model/fp32/resnet50_fp32.dla";

// Matches `input_1` in `resnet50_mtk_int8.tflite`: real-valued range [0,1].
inline constexpr ClassificationQuantParam kInt8InputQuant = {0.00392157f, -128};
// Matches output `activation_49`: scale=1/256, zp=-128, representing softmax probabilities in [0,1].
inline constexpr ClassificationQuantParam kInt8OutputQuant = {0.00390625f, -128};

inline const std::vector<std::string> kDefaultImagePaths = {
    "assets/images/pickup.jpg",
    "assets/images/rhodesian_ridgeback.jpg",
};

inline const std::string kDefaultOutputDir = "outputs/resnet50";

inline ClassificationDemoConfig Config() {
  return ClassificationDemoConfig{
      "resnet50",
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
      ClassificationLabelMapping{kClassCount, kLabelOffset, kOutputTensorSize},
  };
}

}  // namespace mtkdemo::resnet50
