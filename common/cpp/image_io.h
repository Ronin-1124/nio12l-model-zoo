#pragma once

#include <string>
#include <vector>

#include "common/cpp/image_types.h"

namespace mtkdemo {

Image LoadImage(const std::string& image_path);
void SaveImage(const std::string& image_path, const Image& image);
void DrawDetections(Image* image, const std::vector<Detection>& detections,
                    const std::vector<std::string>& labels);

}  // namespace mtkdemo
