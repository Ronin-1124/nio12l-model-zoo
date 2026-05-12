#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cpp/image_io.h"
#include "common/cpp/neuron_runtime.h"
#include "common/cpp/preprocess.h"
#include "common/cpp/yolo26n-pose_postprocess.h"
#include "examples/yolo26n-pose/yolo26n-pose_config.h"

namespace fs = std::filesystem;

namespace {
struct Options {
  std::string model_path;
  std::vector<std::string> image_paths;
  std::string output_dir = mtkdemo::yolo26npose::kDefaultOutputDir;
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
    options.model_path = options.use_fp32 ? mtkdemo::yolo26npose::kDefaultFp32ModelPath
                                          : mtkdemo::yolo26npose::kDefaultInt8ModelPath;
  }
  if (options.image_paths.empty()) {
    options.image_paths.assign(mtkdemo::yolo26npose::kDefaultImagePaths.begin(),
                               mtkdemo::yolo26npose::kDefaultImagePaths.end());
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

void DrawCircle(mtkdemo::Image* image, int cx, int cy, int radius, uint8_t r, uint8_t g, uint8_t b) {
  for (int y = cy - radius; y <= cy + radius; ++y) {
    for (int x = cx - radius; x <= cx + radius; ++x) {
      const int dx = x - cx;
      const int dy = y - cy;
      if (dx * dx + dy * dy <= radius * radius) SetPixelRgb(image, x, y, r, g, b);
    }
  }
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

void DrawPose(mtkdemo::Image* image, const std::vector<mtkdemo::PoseInstance>& poses, float kpt_thr) {
  static constexpr int kEdges[][2] = {
      {5, 7},   {7, 9},   {6, 8},   {8, 10}, {5, 6},  {5, 11}, {6, 12},
      {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16},
  };
  for (const auto& p : poses) {
    const auto& kp = p.keypoints;
    if (kp.size() < static_cast<size_t>(mtkdemo::yolo26npose::kKeypointCount) * 3U) continue;
    for (const auto& e : kEdges) {
      const int a = e[0], b = e[1];
      const float sa = kp[static_cast<size_t>(a) * 3U + 2U];
      const float sb = kp[static_cast<size_t>(b) * 3U + 2U];
      if (sa < kpt_thr || sb < kpt_thr) continue;
      DrawLine(image, static_cast<int>(std::lround(kp[static_cast<size_t>(a) * 3U + 0U])),
               static_cast<int>(std::lround(kp[static_cast<size_t>(a) * 3U + 1U])),
               static_cast<int>(std::lround(kp[static_cast<size_t>(b) * 3U + 0U])),
               static_cast<int>(std::lround(kp[static_cast<size_t>(b) * 3U + 1U])), 0, 255, 255);
    }
    for (int i = 0; i < mtkdemo::yolo26npose::kKeypointCount; ++i) {
      const float s = kp[static_cast<size_t>(i) * 3U + 2U];
      if (s < kpt_thr) continue;
      DrawCircle(image, static_cast<int>(std::lround(kp[static_cast<size_t>(i) * 3U + 0U])),
                 static_cast<int>(std::lround(kp[static_cast<size_t>(i) * 3U + 1U])), 2, 0, 255, 0);
    }
  }
}

void WritePosesJson(const fs::path& path, const std::vector<mtkdemo::PoseInstance>& poses) {
  std::ofstream stream(path);
  if (!stream) throw std::runtime_error("Failed to open poses json: " + path.string());
  stream << "[\n";
  for (size_t i = 0; i < poses.size(); ++i) {
    const auto& p = poses[i];
    stream << "  {\n"
           << "    \"confidence\": " << std::fixed << std::setprecision(4) << p.detection.confidence << ",\n"
           << "    \"x1\": " << p.detection.x1 << ",\n"
           << "    \"y1\": " << p.detection.y1 << ",\n"
           << "    \"x2\": " << p.detection.x2 << ",\n"
           << "    \"y2\": " << p.detection.y2 << ",\n"
           << "    \"keypoints\": [";
    for (size_t k = 0; k < p.keypoints.size(); ++k) {
      stream << std::fixed << std::setprecision(2) << p.keypoints[k];
      if (k + 1 != p.keypoints.size()) stream << ", ";
    }
    stream << "]\n"
           << "  }";
    if (i + 1 != poses.size()) stream << ",";
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
      throw std::runtime_error("Current demo expects one-input nine-output YOLO26n-pose model");
    }
    const size_t input_size = runtime.InputSize(0);
    if (input_size == static_cast<size_t>(mtkdemo::yolo26npose::kInputWidth) *
                          static_cast<size_t>(mtkdemo::yolo26npose::kInputHeight) * 3U) {
      use_int8_io = true;
    } else if (input_size == static_cast<size_t>(mtkdemo::yolo26npose::kInputWidth) *
                                 static_cast<size_t>(mtkdemo::yolo26npose::kInputHeight) * 3U *
                                 sizeof(float)) {
      use_int8_io = false;
    } else {
      throw std::runtime_error("Unsupported model input size for YOLO26n-pose demo");
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
            image, mtkdemo::yolo26npose::kInputWidth, mtkdemo::yolo26npose::kInputHeight,
            mtkdemo::yolo26npose::kInt8InputQuant.scale, mtkdemo::yolo26npose::kInt8InputQuant.zero_point,
            &letterbox);
        input_bytes = ToBytes(input);
      } else {
        const auto input = mtkdemo::PreprocessToFloatRgbNchw(
            image, mtkdemo::yolo26npose::kInputWidth, mtkdemo::yolo26npose::kInputHeight, &letterbox);
        input_bytes = ToBytes(input);
      }

      const fs::path image_stem = fs::path(image_path).stem();
      std::cout << "Image: " << image_path << "\n";

      if (options.skip_runtime) {
        if (runtime.InputSize(0) != input_bytes.size()) {
          throw std::runtime_error("Model input size does not match preprocessing output");
        }
        const fs::path input_bin_path = output_dir / "input_bins" / (image_stem.string() + ".bin");
        mtkdemo::WriteBinaryFile(input_bin_path.string(), input_bytes.data(), input_bytes.size());
        std::cout << "Preprocess output: " << input_bytes.size()
                  << " bytes (matches model input[0])\n";
        std::cout << "Wrote input bin: " << input_bin_path << "\n";
        std::cout << "Skipping NeuronRuntimeV2_run (--skip-runtime).\n";
        continue;
      }
      if (runtime.InputSize(0) != input_bytes.size()) {
        throw std::runtime_error("Model input size does not match preprocessing output");
      }

      std::vector<std::vector<uint8_t>> outputs;
      runtime.Run({input_bytes}, &outputs);
      const std::vector<std::vector<uint8_t>> kpt_outputs = {outputs[0], outputs[1], outputs[2]};
      const std::vector<std::vector<uint8_t>> bbox_outputs = {outputs[3], outputs[4], outputs[5]};
      const std::vector<std::vector<uint8_t>> score_outputs = {outputs[6], outputs[7], outputs[8]};

      std::vector<mtkdemo::PoseInstance> poses;
      if (use_int8_io) {
        std::vector<mtkdemo::YoloV8PoseQuantParam> kpt_q, bbox_q, score_q;
        for (const auto& q : mtkdemo::yolo26npose::kInt8KptQuant) kpt_q.push_back({q.scale, q.zero_point});
        for (const auto& q : mtkdemo::yolo26npose::kInt8BboxQuant)
          bbox_q.push_back({q.scale, q.zero_point});
        for (const auto& q : mtkdemo::yolo26npose::kInt8ScoreQuant)
          score_q.push_back({q.scale, q.zero_point});
        poses = mtkdemo::DecodeYolo26PoseMultiOutputInt8(
            kpt_outputs, kpt_q, score_outputs, score_q, bbox_outputs, bbox_q, letterbox, image.width,
            image.height, mtkdemo::yolo26npose::kConfidenceThreshold, mtkdemo::yolo26npose::kIouThreshold,
            mtkdemo::yolo26npose::kKeypointCount, mtkdemo::yolo26npose::kInputWidth);
      } else {
        poses = mtkdemo::DecodeYolo26PoseMultiOutput(
            kpt_outputs, score_outputs, bbox_outputs, letterbox, image.width, image.height,
            mtkdemo::yolo26npose::kConfidenceThreshold, mtkdemo::yolo26npose::kIouThreshold,
            mtkdemo::yolo26npose::kKeypointCount, mtkdemo::yolo26npose::kInputWidth);
      }

      std::vector<mtkdemo::Detection> dets;
      for (const auto& p : poses) dets.push_back(p.detection);
      mtkdemo::Image vis = image;
      mtkdemo::DrawDetections(&vis, dets, std::vector<std::string>{"person"});
      DrawPose(&vis, poses, mtkdemo::yolo26npose::kKeypointThreshold);
      const fs::path vis_path = vis_dir / (image_stem.string() + "_pose.png");
      mtkdemo::SaveImage(vis_path.string(), vis);
      const fs::path json_path = det_dir / (image_stem.string() + ".json");
      WritePosesJson(json_path, poses);
      std::cout << "Visualized result: " << vis_path << "\n";
      std::cout << "Detections json: " << json_path << "\n";
      std::cout << "Poses: " << poses.size() << "\n";
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "yolo26n-pose_demo failed: " << ex.what() << "\n";
    return 1;
  }
}

