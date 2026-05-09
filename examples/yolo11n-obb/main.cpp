#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cpp/dota_labels.h"
#include "common/cpp/image_io.h"
#include "common/cpp/neuron_runtime.h"
#include "common/cpp/preprocess.h"
#include "common/cpp/yolov8n-obb_postprocess.h"
#include "examples/yolo11n-obb/yolo11n-obb_config.h"

namespace fs = std::filesystem;

namespace {

struct Options {
  std::string model_path;
  std::vector<std::string> image_paths;
  std::string output_dir = mtkdemo::yolo11nobb::kDefaultOutputDir;
  size_t threads = 1;
  size_t backlog = 1;
  bool skip_runtime = false;
  bool use_fp32 = false;
  bool model_overridden = false;
  std::string runtime_options;
};

void PrintUsage(const char* argv0) {
  std::cout << "Usage: " << argv0
            << " [--fp32] [--model path] [--image path ...] [--output-dir dir]"
            << " [--threads n] [--backlog n] [--runtime-options opts] [--skip-runtime]\n";
}

Options ParseArgs(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      options.model_path = argv[++i];
      options.model_overridden = true;
    } else if (arg == "--image" && i + 1 < argc) {
      options.image_paths.push_back(argv[++i]);
    } else if (arg == "--output-dir" && i + 1 < argc) {
      options.output_dir = argv[++i];
    } else if (arg == "--threads" && i + 1 < argc) {
      options.threads = std::stoul(argv[++i]);
    } else if (arg == "--backlog" && i + 1 < argc) {
      options.backlog = std::stoul(argv[++i]);
    } else if (arg == "--runtime-options" && i + 1 < argc) {
      options.runtime_options = argv[++i];
    } else if (arg == "--fp32") {
      options.use_fp32 = true;
    } else if (arg == "--skip-runtime") {
      options.skip_runtime = true;
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }
  if (!options.model_overridden) {
    options.model_path = options.use_fp32 ? mtkdemo::yolo11nobb::kDefaultFp32ModelPath
                                          : mtkdemo::yolo11nobb::kDefaultInt8ModelPath;
  }
  if (options.image_paths.empty()) {
    options.image_paths.assign(mtkdemo::yolo11nobb::kDefaultImagePaths.begin(),
                               mtkdemo::yolo11nobb::kDefaultImagePaths.end());
  }
  return options;
}

template <typename T>
std::vector<uint8_t> ToBytes(const std::vector<T>& values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

inline void SetPixelRgb(mtkdemo::Image* image, int x, int y, uint8_t r, uint8_t g, uint8_t b) {
  if (image == nullptr || image->channels < 3) return;
  if (x < 0 || y < 0 || x >= image->width || y >= image->height) return;
  const size_t off =
      (static_cast<size_t>(y) * static_cast<size_t>(image->width) + static_cast<size_t>(x)) * 3U;
  image->data[off + 0] = r;
  image->data[off + 1] = g;
  image->data[off + 2] = b;
}

void DrawLine(mtkdemo::Image* image, int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b) {
  const int dx = std::abs(x1 - x0);
  const int dy = -std::abs(y1 - y0);
  const int sx = x0 < x1 ? 1 : -1;
  const int sy = y0 < y1 ? 1 : -1;
  int err = dx + dy;
  int x = x0;
  int y = y0;
  while (true) {
    SetPixelRgb(image, x, y, r, g, b);
    if (x == x1 && y == y1) break;
    const int e2 = 2 * err;
    if (e2 >= dy) {
      err += dy;
      x += sx;
    }
    if (e2 <= dx) {
      err += dx;
      y += sy;
    }
  }
}

std::array<std::pair<float, float>, 4> Corners(const mtkdemo::OrientedDetection& d) {
  const float c = std::cos(d.angle);
  const float s = std::sin(d.angle);
  const float hw = d.w * 0.5f;
  const float hh = d.h * 0.5f;
  std::array<std::pair<float, float>, 4> pts = {
      std::pair<float, float>{-hw, -hh}, std::pair<float, float>{hw, -hh},
      std::pair<float, float>{hw, hh}, std::pair<float, float>{-hw, hh}};
  for (auto& p : pts) {
    const float x = p.first;
    const float y = p.second;
    p.first = d.cx + x * c - y * s;
    p.second = d.cy + x * s + y * c;
  }
  return pts;
}

void DrawObbDetections(mtkdemo::Image* image, const std::vector<mtkdemo::OrientedDetection>& detections) {
  for (const auto& d : detections) {
    const auto pts = Corners(d);
    for (int i = 0; i < 4; ++i) {
      const auto& a = pts[static_cast<size_t>(i)];
      const auto& b = pts[static_cast<size_t>((i + 1) % 4)];
      DrawLine(image, static_cast<int>(std::lround(a.first)), static_cast<int>(std::lround(a.second)),
               static_cast<int>(std::lround(b.first)), static_cast<int>(std::lround(b.second)), 255, 0, 0);
    }
  }
}

void WriteObbJson(const fs::path& path, const std::vector<mtkdemo::OrientedDetection>& detections,
                  const std::vector<std::string>& labels) {
  std::ofstream stream(path);
  if (!stream) throw std::runtime_error("Failed to open detections json: " + path.string());
  stream << "[\n";
  for (size_t i = 0; i < detections.size(); ++i) {
    const auto& d = detections[i];
    const std::string label =
        d.class_id >= 0 && static_cast<size_t>(d.class_id) < labels.size() ? labels[d.class_id] : "unknown";
    stream << "  {\n"
           << "    \"class_id\": " << d.class_id << ",\n"
           << "    \"label\": \"" << label << "\",\n"
           << "    \"confidence\": " << std::fixed << std::setprecision(4) << d.confidence << ",\n"
           << "    \"cx\": " << d.cx << ",\n"
           << "    \"cy\": " << d.cy << ",\n"
           << "    \"w\": " << d.w << ",\n"
           << "    \"h\": " << d.h << ",\n"
           << "    \"angle_rad\": " << d.angle << "\n"
           << "  }";
    if (i + 1 != detections.size()) stream << ",";
    stream << "\n";
  }
  stream << "]\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseArgs(argc, argv);
    const fs::path output_dir = options.output_dir;
    const fs::path vis_dir = output_dir / "vis";
    const fs::path det_dir = output_dir / "detections";
    fs::create_directories(vis_dir);
    fs::create_directories(det_dir);

    bool use_int8_io = !options.use_fp32;
    mtkdemo::NeuronRuntimeRunner runtime;
    runtime.Load(options.model_path, options.threads, options.backlog, options.runtime_options);
    if (runtime.InputCount() != 1 || runtime.OutputCount() != 9) {
      throw std::runtime_error("Current demo expects one-input nine-output YOLO11n-OBB model");
    }
    const size_t input_size = runtime.InputSize(0);
    if (input_size == static_cast<size_t>(mtkdemo::yolo11nobb::kInputWidth) *
                          static_cast<size_t>(mtkdemo::yolo11nobb::kInputHeight) * 3U) {
      use_int8_io = true;
    } else if (input_size == static_cast<size_t>(mtkdemo::yolo11nobb::kInputWidth) *
                                 static_cast<size_t>(mtkdemo::yolo11nobb::kInputHeight) * 3U *
                                 sizeof(float)) {
      use_int8_io = false;
    } else {
      throw std::runtime_error("Unsupported model input size for YOLO11n-OBB demo");
    }

    std::cout << "Model: " << options.model_path << "\n";
    std::cout << "Input dtype: " << (use_int8_io ? "int8" : "fp32") << "\n";
    if (options.skip_runtime) mtkdemo::PrintNeuronRuntimeIoSummary(std::cout, runtime);

    for (const std::string& image_path : options.image_paths) {
      mtkdemo::Image image = mtkdemo::LoadImage(image_path);
      mtkdemo::LetterboxInfo letterbox;
      std::vector<uint8_t> input_bytes;
      if (use_int8_io) {
        const auto input = mtkdemo::PreprocessToInt8RgbNchw(
            image, mtkdemo::yolo11nobb::kInputWidth, mtkdemo::yolo11nobb::kInputHeight,
            mtkdemo::yolo11nobb::kInt8InputQuant.scale, mtkdemo::yolo11nobb::kInt8InputQuant.zero_point,
            &letterbox);
        input_bytes = ToBytes(input);
      } else {
        const auto input = mtkdemo::PreprocessToFloatRgbNchw(
            image, mtkdemo::yolo11nobb::kInputWidth, mtkdemo::yolo11nobb::kInputHeight, &letterbox);
        input_bytes = ToBytes(input);
      }

      const fs::path image_stem = fs::path(image_path).stem();
      std::cout << "Image: " << image_path << "\n";
      if (options.skip_runtime) {
        if (runtime.InputSize(0) != input_bytes.size()) {
          throw std::runtime_error("Model input size does not match preprocessing output");
        }
        std::cout << "Preprocess output: " << input_bytes.size()
                  << " bytes (matches model input[0])\n";
        std::cout << "Skipping NeuronRuntimeV2_run (--skip-runtime).\n";
        continue;
      }
      if (runtime.InputSize(0) != input_bytes.size()) {
        throw std::runtime_error("Model input size does not match preprocessing output");
      }

      std::vector<std::vector<uint8_t>> outputs;
      runtime.Run({input_bytes}, &outputs);
      const std::vector<std::vector<uint8_t>> angle_outputs = {outputs[0], outputs[1], outputs[2]};
      const std::vector<std::vector<uint8_t>> cls_outputs = {outputs[3], outputs[4], outputs[5]};
      const std::vector<std::vector<uint8_t>> bbox_outputs = {outputs[6], outputs[7], outputs[8]};

      std::vector<mtkdemo::OrientedDetection> detections;
      if (use_int8_io) {
        std::vector<mtkdemo::YoloObbQuantParam> angle_q, cls_q, bbox_q;
        for (const auto& q : mtkdemo::yolo11nobb::kInt8AngleQuant) angle_q.push_back({q.scale, q.zero_point});
        for (const auto& q : mtkdemo::yolo11nobb::kInt8ClsQuant) cls_q.push_back({q.scale, q.zero_point});
        for (const auto& q : mtkdemo::yolo11nobb::kInt8BboxQuant) bbox_q.push_back({q.scale, q.zero_point});
        detections = mtkdemo::DecodeYoloV8ObbMultiOutputInt8(
            angle_outputs, angle_q, cls_outputs, cls_q, bbox_outputs, bbox_q, letterbox, image.width,
            image.height, mtkdemo::yolo11nobb::kConfidenceThreshold, mtkdemo::yolo11nobb::kIouThreshold,
            mtkdemo::yolo11nobb::kClassCount, mtkdemo::yolo11nobb::kRegMax, mtkdemo::yolo11nobb::kInputWidth);
      } else {
        detections = mtkdemo::DecodeYoloV8ObbMultiOutput(
            angle_outputs, cls_outputs, bbox_outputs, letterbox, image.width, image.height,
            mtkdemo::yolo11nobb::kConfidenceThreshold, mtkdemo::yolo11nobb::kIouThreshold,
            mtkdemo::yolo11nobb::kClassCount, mtkdemo::yolo11nobb::kRegMax, mtkdemo::yolo11nobb::kInputWidth);
      }

      mtkdemo::Image vis = image;
      DrawObbDetections(&vis, detections);
      const fs::path vis_path = vis_dir / (image_stem.string() + "_obb.png");
      mtkdemo::SaveImage(vis_path.string(), vis);
      const fs::path json_path = det_dir / (image_stem.string() + ".json");
      WriteObbJson(json_path, detections, mtkdemo::DotaLabels());
      std::cout << "Visualized result: " << vis_path << "\n";
      std::cout << "Detections json: " << json_path << "\n";
      std::cout << "Detections: " << detections.size() << "\n";
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "yolo11n-obb_demo failed: " << ex.what() << "\n";
    return 1;
  }
}
