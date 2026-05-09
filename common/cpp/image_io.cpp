#include "common/cpp/image_io.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <stdexcept>

#include "third_party/stb/stb_image.h"
#include "third_party/stb/stb_image_write.h"

namespace mtkdemo {
namespace {

std::array<uint8_t, 3> ColorForClass(int class_id) {
  static constexpr std::array<std::array<uint8_t, 3>, 6> kPalette = {{
      {255, 56, 56},
      {255, 157, 151},
      {255, 112, 31},
      {255, 178, 29},
      {207, 210, 49},
      {72, 249, 10},
  }};
  return kPalette[static_cast<size_t>(class_id >= 0 ? class_id : 0) % kPalette.size()];
}

void SetPixel(Image* image, int x, int y, uint8_t r, uint8_t g, uint8_t b) {
  if (x < 0 || y < 0 || x >= image->width || y >= image->height) {
    return;
  }
  const size_t offset =
      (static_cast<size_t>(y) * static_cast<size_t>(image->width) + static_cast<size_t>(x)) * 3U;
  image->data[offset + 0] = r;
  image->data[offset + 1] = g;
  image->data[offset + 2] = b;
}

void DrawRect(Image* image, int x1, int y1, int x2, int y2, uint8_t r, uint8_t g, uint8_t b,
              int thickness) {
  for (int t = 0; t < thickness; ++t) {
    for (int x = x1; x <= x2; ++x) {
      SetPixel(image, x, y1 + t, r, g, b);
      SetPixel(image, x, y2 - t, r, g, b);
    }
    for (int y = y1; y <= y2; ++y) {
      SetPixel(image, x1 + t, y, r, g, b);
      SetPixel(image, x2 - t, y, r, g, b);
    }
  }
}

std::array<uint8_t, 5> Glyph5x7(char ch) {
  switch (ch) {
    case 'A': return {0x1E, 0x05, 0x05, 0x1E, 0x00};
    case 'B': return {0x1F, 0x15, 0x15, 0x0A, 0x00};
    case 'C': return {0x0E, 0x11, 0x11, 0x11, 0x00};
    case 'D': return {0x1F, 0x11, 0x11, 0x0E, 0x00};
    case 'E': return {0x1F, 0x15, 0x15, 0x11, 0x00};
    case 'F': return {0x1F, 0x05, 0x05, 0x01, 0x00};
    case 'G': return {0x0E, 0x11, 0x15, 0x1D, 0x00};
    case 'H': return {0x1F, 0x04, 0x04, 0x1F, 0x00};
    case 'I': return {0x11, 0x1F, 0x11, 0x00, 0x00};
    case 'J': return {0x08, 0x10, 0x10, 0x0F, 0x00};
    case 'K': return {0x1F, 0x04, 0x0A, 0x11, 0x00};
    case 'L': return {0x1F, 0x10, 0x10, 0x10, 0x00};
    case 'M': return {0x1F, 0x02, 0x04, 0x02, 0x1F};
    case 'N': return {0x1F, 0x02, 0x04, 0x1F, 0x00};
    case 'O': return {0x0E, 0x11, 0x11, 0x0E, 0x00};
    case 'P': return {0x1F, 0x05, 0x05, 0x02, 0x00};
    case 'Q': return {0x0E, 0x11, 0x19, 0x1E, 0x00};
    case 'R': return {0x1F, 0x05, 0x0D, 0x12, 0x00};
    case 'S': return {0x12, 0x15, 0x15, 0x09, 0x00};
    case 'T': return {0x01, 0x1F, 0x01, 0x00, 0x00};
    case 'U': return {0x0F, 0x10, 0x10, 0x0F, 0x00};
    case 'V': return {0x07, 0x08, 0x10, 0x08, 0x07};
    case 'W': return {0x1F, 0x08, 0x04, 0x08, 0x1F};
    case 'X': return {0x1B, 0x04, 0x04, 0x1B, 0x00};
    case 'Y': return {0x03, 0x04, 0x18, 0x04, 0x03};
    case 'Z': return {0x19, 0x15, 0x13, 0x00, 0x00};
    case '0': return {0x0E, 0x11, 0x11, 0x0E, 0x00};
    case '1': return {0x12, 0x1F, 0x10, 0x00, 0x00};
    case '2': return {0x19, 0x15, 0x15, 0x12, 0x00};
    case '3': return {0x11, 0x15, 0x15, 0x0A, 0x00};
    case '4': return {0x07, 0x04, 0x04, 0x1F, 0x00};
    case '5': return {0x17, 0x15, 0x15, 0x09, 0x00};
    case '6': return {0x0E, 0x15, 0x15, 0x08, 0x00};
    case '7': return {0x01, 0x01, 0x1D, 0x03, 0x00};
    case '8': return {0x0A, 0x15, 0x15, 0x0A, 0x00};
    case '9': return {0x02, 0x15, 0x15, 0x0E, 0x00};
    case '.': return {0x10, 0x00, 0x00, 0x00, 0x00};
    case '-': return {0x04, 0x04, 0x04, 0x04, 0x00};
    case '_': return {0x10, 0x10, 0x10, 0x10, 0x00};
    case ' ': return {0x00, 0x00, 0x00, 0x00, 0x00};
    default: return {0x1F, 0x11, 0x11, 0x1F, 0x00};  // Unknown -> box.
  }
}

void DrawText(Image* image, int x, int y, const std::string& text, uint8_t r, uint8_t g, uint8_t b) {
  int cursor_x = x;
  for (char ch : text) {
    const char up = (ch >= 'a' && ch <= 'z') ? static_cast<char>(ch - 'a' + 'A') : ch;
    const auto glyph = Glyph5x7(up);
    for (int col = 0; col < 5; ++col) {
      for (int row = 0; row < 7; ++row) {
        if ((glyph[col] >> row) & 0x01U) {
          SetPixel(image, cursor_x + col, y + row, r, g, b);
        }
      }
    }
    cursor_x += 6;  // 5 px glyph + 1 px spacing.
  }
}

}  // namespace

Image LoadImage(const std::string& image_path) {
  int width = 0;
  int height = 0;
  int channels = 0;
  stbi_uc* pixels = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
  if (pixels == nullptr) {
    throw std::runtime_error("Failed to load image: " + image_path);
  }

  Image image;
  image.width = width;
  image.height = height;
  image.channels = 3;
  image.data.assign(pixels, pixels + static_cast<size_t>(width) * static_cast<size_t>(height) * 3U);
  stbi_image_free(pixels);
  return image;
}

void SaveImage(const std::string& image_path, const Image& image) {
  if (image.channels != 3) {
    throw std::runtime_error("Only 3-channel RGB output is supported");
  }

  const int stride = image.width * image.channels;
  const int ok = stbi_write_png(image_path.c_str(), image.width, image.height, image.channels,
                                image.data.data(), stride);
  if (ok == 0) {
    throw std::runtime_error("Failed to save image: " + image_path);
  }
}

void DrawDetections(Image* image, const std::vector<Detection>& detections,
                    const std::vector<std::string>& labels) {
  for (const Detection& detection : detections) {
    const int x1 = std::clamp(static_cast<int>(detection.x1), 0, image->width - 1);
    const int y1 = std::clamp(static_cast<int>(detection.y1), 0, image->height - 1);
    const int x2 = std::clamp(static_cast<int>(detection.x2), 0, image->width - 1);
    const int y2 = std::clamp(static_cast<int>(detection.y2), 0, image->height - 1);
    const auto color = ColorForClass(detection.class_id);
    DrawRect(image, x1, y1, x2, y2, color[0], color[1], color[2], 2);

    std::string label = "unknown";
    if (detection.class_id >= 0 && static_cast<size_t>(detection.class_id) < labels.size()) {
      label = labels[static_cast<size_t>(detection.class_id)];
    }
    std::ostringstream text_stream;
    text_stream << label << " " << std::fixed << std::setprecision(2) << detection.confidence;
    const std::string text = text_stream.str();

    const int text_w = static_cast<int>(text.size()) * 6;
    const int text_h = 7;
    const int pad = 2;
    const int bar_height = text_h + pad * 2;
    const int bar_width = std::min(image->width - x1, text_w + pad * 2);
    const int bar_top = std::max(0, y1 - bar_height);
    for (int y = bar_top; y < y1; ++y) {
      for (int x = x1; x < x1 + bar_width; ++x) {
        SetPixel(image, x, y, color[0], color[1], color[2]);
      }
    }
    DrawText(image, x1 + pad, bar_top + pad, text, 255, 255, 255);
  }
}

}  // namespace mtkdemo
