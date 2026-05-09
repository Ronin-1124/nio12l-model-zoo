#include "common/cpp/dota_labels.h"

namespace mtkdemo {

const std::vector<std::string>& DotaLabels() {
  static const std::vector<std::string> labels = {
      "plane",        "ship",       "storage-tank", "baseball-diamond", "tennis-court",
      "basketball-court", "ground-track-field", "harbor", "bridge",       "large-vehicle",
      "small-vehicle", "helicopter", "roundabout", "soccer-ball-field", "swimming-pool",
  };
  return labels;
}

}  // namespace mtkdemo
