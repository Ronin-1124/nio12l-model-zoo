#pragma once

#include <string>
#include <vector>

#include "common/cpp/ssd_mobilenet_v2_postprocess.h"

namespace mtkdemo::ssd_mobilenet_v2 {

struct QuantParam {
  float scale = 1.0f;
  int zero_point = 0;
};

inline constexpr int kInputSize = 300;
// SSD raw_detection_scores use per-class independent sigmoid (non-mutually-exclusive), unlike YOLO obj x cls
// This creates a flatter score distribution where negative anchors may keep 0.3~0.5 scores, so default threshold is higher than YOLO (0.25).
inline constexpr float kScoreThreshold = 0.5f;
inline constexpr float kIouThreshold = 0.45f;
inline constexpr size_t kTopK = 100;

inline const std::string kDefaultInt8ModelPath =
    "examples/ssd_mobilenet_v2/model/int8/ssd_mobilenet_v2_int8.dla";
inline const std::string kDefaultFp32ModelPath =
    "examples/ssd_mobilenet_v2/model/fp32/ssd_mobilenet_v2_fp32.dla";

/// Matches `preprocessed_input` in `ssd_mobilenet_v2_mtk_int8.tflite`: real-valued input range is about [0,1].
inline constexpr QuantParam kInt8InputQuant = {0.00392157f, -128};
/// Matches `raw_detection_boxes` / `raw_detection_scores` in TFLite.
inline constexpr QuantParam kInt8BoxesQuant = {0.00592771f, -83};
inline constexpr QuantParam kInt8ScoresQuant = {0.00390625f, -128};

inline SsdQuantParam BoxesQuant() {
  return SsdQuantParam{kInt8BoxesQuant.scale, kInt8BoxesQuant.zero_point};
}
inline SsdQuantParam ScoresQuant() {
  return SsdQuantParam{kInt8ScoresQuant.scale, kInt8ScoresQuant.zero_point};
}

inline const std::vector<std::string> kDefaultImagePaths = {
    "assets/images/bus.jpg",
    "assets/images/zidane.jpg",
};

inline const std::string kDefaultOutputDir = "outputs/ssd_mobilenet_v2";

}  // namespace mtkdemo::ssd_mobilenet_v2
