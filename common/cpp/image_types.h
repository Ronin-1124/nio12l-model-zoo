#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mtkdemo {

struct Image {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::vector<uint8_t> data;
};

struct Detection {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float confidence = 0.0f;
  int class_id = -1;
};

struct LetterboxInfo {
  int resized_width = 0;
  int resized_height = 0;
  int pad_x = 0;
  int pad_y = 0;
  float scale = 1.0f;
};

}  // namespace mtkdemo
