#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/cpp/image_types.h"

namespace mtkdemo {

std::vector<float> PreprocessToFloatRgb(const Image& image, int target_width,
                                        int target_height, LetterboxInfo* info);
std::vector<float> PreprocessToFloatRgbNchw(const Image& image, int target_width,
                                            int target_height, LetterboxInfo* info);
std::vector<int8_t> PreprocessToInt8RgbNchw(const Image& image, int target_width,
                                            int target_height, float scale, int zero_point,
                                            LetterboxInfo* info);

/// SSD MobileNet V2 trimmed graph: stretch-resize to 300x300, RGB float NCHW, normalize with x/127.5-1.
/// `LoadImage` decodes with stb as RGB; if your pipeline is BGR, swap R/B before calling.
std::vector<float> PreprocessSsdMobilenetV2FloatNchw(const Image& image);
/// SSD MobileNet V2 TFLite/DLA variant: stretch-resize to 300x300, RGB float NCHW, normalize with x/255.
std::vector<float> PreprocessSsdMobilenetV2Float01Nchw(const Image& image);
/// TFLite/DLA INT8 input quantization scale=1/255, zero_point=-128 (real-value range [0,1]).
std::vector<int8_t> PreprocessSsdMobilenetV2Int8Nchw(const Image& image, float scale,
                                                     int zero_point);

void WriteBinaryFile(const std::string& path, const void* data, size_t size);

}  // namespace mtkdemo
