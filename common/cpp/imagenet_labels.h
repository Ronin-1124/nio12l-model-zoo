#pragma once

#include <string>
#include <vector>

namespace mtkdemo {

// ImageNet labels for MobileNet-style 1001-class outputs.
// Index 0 is background, indexes 1..1000 are ILSVRC ImageNet classes.
const std::vector<std::string>& ImagenetLabels();

}  // namespace mtkdemo
