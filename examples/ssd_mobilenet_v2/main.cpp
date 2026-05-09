#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>

#include "common/cpp/coco91_labels.h"
#include "common/cpp/image_io.h"
#include "common/cpp/neuron_runtime.h"
#include "common/cpp/preprocess.h"
#include "common/cpp/ssd_mobilenet_v2_postprocess.h"
#include "examples/ssd_mobilenet_v2/ssd_mobilenet_v2_config.h"

namespace fs = std::filesystem;

namespace {

struct Options {
  std::string model_path;
  std::vector<std::string> image_paths;
  std::string output_dir = mtkdemo::ssd_mobilenet_v2::kDefaultOutputDir;
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
    options.model_path = options.use_fp32 ? mtkdemo::ssd_mobilenet_v2::kDefaultFp32ModelPath
                                          : mtkdemo::ssd_mobilenet_v2::kDefaultInt8ModelPath;
  }
  if (options.image_paths.empty()) {
    options.image_paths.assign(mtkdemo::ssd_mobilenet_v2::kDefaultImagePaths.begin(),
                               mtkdemo::ssd_mobilenet_v2::kDefaultImagePaths.end());
  }

  return options;
}

template <typename T>
std::vector<uint8_t> ToBytes(const std::vector<T>& values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

void WriteDetectionsJson(const fs::path& path, const std::vector<mtkdemo::Detection>& detections,
                         const std::vector<std::string>& labels) {
  std::ofstream stream(path);
  if (!stream) {
    throw std::runtime_error("Failed to open detections json: " + path.string());
  }

  stream << "[\n";
  for (size_t i = 0; i < detections.size(); ++i) {
    const auto& det = detections[i];
    const std::string label =
        det.class_id >= 0 && static_cast<size_t>(det.class_id) < labels.size() ? labels[det.class_id]
                                                                                : "unknown";
    stream << "  {\n"
           << "    \"class_id\": " << det.class_id << ",\n"
           << "    \"label\": \"" << label << "\",\n"
           << "    \"confidence\": " << std::fixed << std::setprecision(4) << det.confidence << ",\n"
           << "    \"x1\": " << det.x1 << ",\n"
           << "    \"y1\": " << det.y1 << ",\n"
           << "    \"x2\": " << det.x2 << ",\n"
           << "    \"y2\": " << det.y2 << "\n"
           << "  }";
    if (i + 1 != detections.size()) {
      stream << ",";
    }
    stream << "\n";
  }
  stream << "]\n";
}

void PickBoxesAndScores(const std::vector<std::vector<uint8_t>>& outputs,
                        bool int8_output,
                        const std::vector<uint8_t>** boxes_out,
                        const std::vector<uint8_t>** scores_out) {
  constexpr size_t kBoxValues = 4;
  constexpr size_t kScoreClasses = 91;
  if (outputs.size() != 2) {
    throw std::runtime_error("SSD demo expects exactly two outputs (raw_detection_boxes, "
                             "raw_detection_scores)");
  }
  const size_t elem_size = int8_output ? sizeof(int8_t) : sizeof(float);

  for (size_t box_idx = 0; box_idx < outputs.size(); ++box_idx) {
    const size_t score_idx = 1 - box_idx;
    const size_t box_bytes = outputs[box_idx].size();
    const size_t score_bytes = outputs[score_idx].size();

    if (box_bytes % elem_size != 0 || score_bytes % elem_size != 0) {
      continue;
    }
    const size_t box_elems = box_bytes / elem_size;
    const size_t score_elems = score_bytes / elem_size;
    if (box_elems % kBoxValues != 0 || score_elems % kScoreClasses != 0) {
      continue;
    }
    const size_t box_anchors = box_elems / kBoxValues;
    const size_t score_anchors = score_elems / kScoreClasses;
    if (box_anchors == 0 || box_anchors != score_anchors) {
      continue;
    }

    *boxes_out = &outputs[box_idx];
    *scores_out = &outputs[score_idx];
    return;
  }

  throw std::runtime_error(
      "SSD output tensors do not match expected shapes [1,N,4] and [1,N,91] for current dtype");
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

    const size_t kInputBytesInt8 =
        1ULL * 3ULL * static_cast<size_t>(mtkdemo::ssd_mobilenet_v2::kInputSize) *
        static_cast<size_t>(mtkdemo::ssd_mobilenet_v2::kInputSize);
    const size_t kInputBytesFp32 = kInputBytesInt8 * sizeof(float);

    mtkdemo::NeuronRuntimeRunner runtime;
    runtime.Load(options.model_path, options.threads, options.backlog, options.runtime_options);
    if (runtime.InputCount() != 1) {
      throw std::runtime_error("SSD demo expects a single-input model");
    }

    bool use_int8_io = !options.use_fp32;
    const size_t input_size = runtime.InputSize(0);
    if (input_size == kInputBytesInt8) {
      use_int8_io = true;
    } else if (input_size == kInputBytesFp32) {
      use_int8_io = false;
    } else {
      throw std::runtime_error("Unsupported model input size for SSD MobileNet V2 demo: " +
                               std::to_string(input_size));
    }

    std::cout << "Model: " << options.model_path << "\n";
    std::cout << "Input dtype: " << (use_int8_io ? "int8" : "fp32") << "\n";
    if (options.skip_runtime) {
      mtkdemo::PrintNeuronRuntimeIoSummary(std::cout, runtime);
    }

    for (const std::string& image_path : options.image_paths) {
      mtkdemo::Image image = mtkdemo::LoadImage(image_path);

      std::vector<uint8_t> input_bytes;
      if (use_int8_io) {
        const std::vector<int8_t> input_tensor = mtkdemo::PreprocessSsdMobilenetV2Int8Nchw(
            image, mtkdemo::ssd_mobilenet_v2::kInt8InputQuant.scale,
            mtkdemo::ssd_mobilenet_v2::kInt8InputQuant.zero_point);
        input_bytes = ToBytes(input_tensor);
      } else {
        const std::vector<float> input_tensor = mtkdemo::PreprocessSsdMobilenetV2Float01Nchw(image);
        input_bytes = ToBytes(input_tensor);
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
        throw std::runtime_error("Model input size does not match current preprocessing output");
      }

      std::vector<std::vector<uint8_t>> outputs;
      runtime.Run({input_bytes}, &outputs);

      const std::vector<uint8_t>* boxes_buf = nullptr;
      const std::vector<uint8_t>* scores_buf = nullptr;
      PickBoxesAndScores(outputs, use_int8_io, &boxes_buf, &scores_buf);

      std::vector<mtkdemo::Detection> detections;
      if (use_int8_io) {
        detections = mtkdemo::DecodeSsdMobilenetV2RawInt8(
            *boxes_buf, *scores_buf, mtkdemo::ssd_mobilenet_v2::BoxesQuant(),
            mtkdemo::ssd_mobilenet_v2::ScoresQuant(), image.width, image.height,
            mtkdemo::ssd_mobilenet_v2::kScoreThreshold,
            mtkdemo::ssd_mobilenet_v2::kIouThreshold, mtkdemo::ssd_mobilenet_v2::kTopK);
      } else {
        detections = mtkdemo::DecodeSsdMobilenetV2Raw(
            *boxes_buf, *scores_buf, image.width, image.height,
            mtkdemo::ssd_mobilenet_v2::kScoreThreshold,
            mtkdemo::ssd_mobilenet_v2::kIouThreshold, mtkdemo::ssd_mobilenet_v2::kTopK);
      }

      mtkdemo::Image visualized = image;
      const auto& labels = mtkdemo::Coco91Labels();
      mtkdemo::DrawDetections(&visualized, detections, labels);
      const fs::path vis_path = vis_dir / (image_stem.string() + "_det.png");
      mtkdemo::SaveImage(vis_path.string(), visualized);

      const fs::path json_path = det_dir / (image_stem.string() + ".json");
      WriteDetectionsJson(json_path, detections, labels);

      std::cout << "Visualized result: " << vis_path << "\n";
      std::cout << "Detections json: " << json_path << "\n";
      std::cout << "Detections: " << detections.size() << "\n";
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "ssd_mobilenet_v2_demo failed: " << ex.what() << "\n";
    return 1;
  }
}
