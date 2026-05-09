#include "common/cpp/preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace mtkdemo {
namespace {

inline uint8_t PixelAt(const Image& image, int x, int y, int channel) {
  const size_t index =
      (static_cast<size_t>(y) * static_cast<size_t>(image.width) + static_cast<size_t>(x)) * 3U +
      static_cast<size_t>(channel);
  return image.data[index];
}

}  // namespace

std::vector<float> PreprocessToFloatRgb(const Image& image, int target_width, int target_height,
                                        LetterboxInfo* info) {
  if (image.channels != 3) {
    throw std::runtime_error("Only 3-channel RGB input is supported");
  }

  const float scale =
      std::min(static_cast<float>(target_width) / static_cast<float>(image.width),
               static_cast<float>(target_height) / static_cast<float>(image.height));
  const int resized_width = std::max(1, static_cast<int>(std::round(image.width * scale)));
  const int resized_height = std::max(1, static_cast<int>(std::round(image.height * scale)));
  const int pad_x = (target_width - resized_width) / 2;
  const int pad_y = (target_height - resized_height) / 2;

  std::vector<float> output(static_cast<size_t>(target_width) * static_cast<size_t>(target_height) *
                                3U,
                            114.0f / 255.0f);

  for (int y = 0; y < resized_height; ++y) {
    const float src_y = (static_cast<float>(y) + 0.5f) / scale - 0.5f;
    const int y0 = std::clamp(static_cast<int>(std::floor(src_y)), 0, image.height - 1);
    const int y1 = std::clamp(y0 + 1, 0, image.height - 1);
    const float ly = src_y - std::floor(src_y);

    for (int x = 0; x < resized_width; ++x) {
      const float src_x = (static_cast<float>(x) + 0.5f) / scale - 0.5f;
      const int x0 = std::clamp(static_cast<int>(std::floor(src_x)), 0, image.width - 1);
      const int x1 = std::clamp(x0 + 1, 0, image.width - 1);
      const float lx = src_x - std::floor(src_x);

      const int dst_x = x + pad_x;
      const int dst_y = y + pad_y;
      const size_t dst_index =
          (static_cast<size_t>(dst_y) * static_cast<size_t>(target_width) + static_cast<size_t>(dst_x)) * 3U;

      for (int c = 0; c < 3; ++c) {
        const float v00 = static_cast<float>(PixelAt(image, x0, y0, c));
        const float v01 = static_cast<float>(PixelAt(image, x1, y0, c));
        const float v10 = static_cast<float>(PixelAt(image, x0, y1, c));
        const float v11 = static_cast<float>(PixelAt(image, x1, y1, c));
        const float top = v00 + (v01 - v00) * lx;
        const float bottom = v10 + (v11 - v10) * lx;
        output[dst_index + static_cast<size_t>(c)] = (top + (bottom - top) * ly) / 255.0f;
      }
    }
  }

  if (info != nullptr) {
    info->resized_width = resized_width;
    info->resized_height = resized_height;
    info->pad_x = pad_x;
    info->pad_y = pad_y;
    info->scale = scale;
  }

  return output;
}

std::vector<float> PreprocessToFloatRgbNchw(const Image& image, int target_width,
                                            int target_height, LetterboxInfo* info) {
  const std::vector<float> nhwc = PreprocessToFloatRgb(image, target_width, target_height, info);
  std::vector<float> nchw(nhwc.size());
  const size_t plane_size = static_cast<size_t>(target_width) * static_cast<size_t>(target_height);

  for (int y = 0; y < target_height; ++y) {
    for (int x = 0; x < target_width; ++x) {
      const size_t nhwc_index =
          (static_cast<size_t>(y) * static_cast<size_t>(target_width) + static_cast<size_t>(x)) * 3U;
      const size_t spatial_index =
          static_cast<size_t>(y) * static_cast<size_t>(target_width) + static_cast<size_t>(x);
      for (size_t c = 0; c < 3U; ++c) {
        nchw[c * plane_size + spatial_index] = nhwc[nhwc_index + c];
      }
    }
  }

  return nchw;
}

std::vector<int8_t> PreprocessToInt8RgbNchw(const Image& image, int target_width,
                                            int target_height, float scale, int zero_point,
                                            LetterboxInfo* info) {
  if (scale <= 0.0f) {
    throw std::runtime_error("INT8 preprocess scale must be positive");
  }

  const std::vector<float> nchw = PreprocessToFloatRgbNchw(image, target_width, target_height, info);
  std::vector<int8_t> quantized(nchw.size());
  for (size_t i = 0; i < nchw.size(); ++i) {
    const float q = std::round(nchw[i] / scale) + static_cast<float>(zero_point);
    const int q_int = std::clamp(static_cast<int>(q), -128, 127);
    quantized[i] = static_cast<int8_t>(q_int);
  }
  return quantized;
}

std::vector<float> PreprocessSsdMobilenetV2FloatNchw(const Image& image) {
  std::vector<float> nchw = PreprocessSsdMobilenetV2Float01Nchw(image);
  for (float& v : nchw) {
    v = v * 2.0f - 1.0f;
  }
  return nchw;
}

std::vector<float> PreprocessSsdMobilenetV2Float01Nchw(const Image& image) {
  constexpr int kTargetW = 300;
  constexpr int kTargetH = 300;
  if (image.channels != 3) {
    throw std::runtime_error("SSD preprocess: only 3-channel RGB input is supported");
  }

  const float src_w = static_cast<float>(image.width);
  const float src_h = static_cast<float>(image.height);
  const size_t plane = static_cast<size_t>(kTargetW) * static_cast<size_t>(kTargetH);
  std::vector<float> nchw(3U * plane);

  for (int dy = 0; dy < kTargetH; ++dy) {
    for (int dx = 0; dx < kTargetW; ++dx) {
      const float sx = (static_cast<float>(dx) + 0.5f) * src_w / static_cast<float>(kTargetW) - 0.5f;
      const float sy = (static_cast<float>(dy) + 0.5f) * src_h / static_cast<float>(kTargetH) - 0.5f;

      const int x0 = std::clamp(static_cast<int>(std::floor(sx)), 0, image.width - 1);
      const int x1 = std::clamp(x0 + 1, 0, image.width - 1);
      const int y0 = std::clamp(static_cast<int>(std::floor(sy)), 0, image.height - 1);
      const int y1 = std::clamp(y0 + 1, 0, image.height - 1);
      const float lx = sx - std::floor(sx);
      const float ly = sy - std::floor(sy);

      const size_t spatial_index =
          static_cast<size_t>(dy) * static_cast<size_t>(kTargetW) + static_cast<size_t>(dx);

      for (int c = 0; c < 3; ++c) {
        const float v00 = static_cast<float>(PixelAt(image, x0, y0, c));
        const float v01 = static_cast<float>(PixelAt(image, x1, y0, c));
        const float v10 = static_cast<float>(PixelAt(image, x0, y1, c));
        const float v11 = static_cast<float>(PixelAt(image, x1, y1, c));
        const float top = v00 + (v01 - v00) * lx;
        const float bottom = v10 + (v11 - v10) * lx;
        const float interp = top + (bottom - top) * ly;
        nchw[static_cast<size_t>(c) * plane + spatial_index] = interp / 255.0f;
      }
    }
  }

  return nchw;
}

std::vector<int8_t> PreprocessSsdMobilenetV2Int8Nchw(const Image& image, float scale,
                                                   int zero_point) {
  if (scale <= 0.0f) {
    throw std::runtime_error("SSD INT8 preprocess: scale must be positive");
  }
  const std::vector<float> nchw = PreprocessSsdMobilenetV2Float01Nchw(image);
  std::vector<int8_t> quantized(nchw.size());
  for (size_t i = 0; i < nchw.size(); ++i) {
    const float q = std::round(nchw[i] / scale) + static_cast<float>(zero_point);
    const int q_int = std::clamp(static_cast<int>(q), -128, 127);
    quantized[i] = static_cast<int8_t>(q_int);
  }
  return quantized;
}

void WriteBinaryFile(const std::string& path, const void* data, size_t size) {
  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  std::ofstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("Failed to open binary file for writing: " + path);
  }
  stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
  if (!stream) {
    throw std::runtime_error("Failed to write binary file: " + path);
  }
}

}  // namespace mtkdemo
