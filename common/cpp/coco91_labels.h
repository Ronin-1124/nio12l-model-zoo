#pragma once

#include <string>
#include <vector>

namespace mtkdemo {

/// TensorFlow `mscoco_complete_label_map.pbtxt` order: index 0..90 maps to id 0..90 (including background).
const std::vector<std::string>& Coco91Labels();

}  // namespace mtkdemo
