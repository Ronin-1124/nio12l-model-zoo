#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cpp/coco_labels.h"
#include "common/cpp/image_io.h"
#include "common/cpp/neuron_runtime.h"
#include "common/cpp/preprocess.h"
#include "common/cpp/yolov8_postprocess.h"
#include "examples/yolov8n/yolov8n_config.h"

namespace fs = std::filesystem;

namespace {

struct Options {
  std::string model_path;
  std::vector<std::string> image_paths;
  std::string output_dir = mtkdemo::yolov8n::kDefaultOutputDir;
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
    options.model_path = options.use_fp32 ? mtkdemo::yolov8n::kDefaultFp32ModelPath
                                          : mtkdemo::yolov8n::kDefaultInt8ModelPath;
  }
  if (options.image_paths.empty()) {
    options.image_paths.assign(mtkdemo::yolov8n::kDefaultImagePaths.begin(),
                               mtkdemo::yolov8n::kDefaultImagePaths.end());
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
    if (runtime.InputCount() != 1 || runtime.OutputCount() != 3) {
      throw std::runtime_error("Current demo expects a single-input, three-output YOLOv8n model");
    }
    const size_t input_size = runtime.InputSize(0);
    if (input_size == static_cast<size_t>(mtkdemo::yolov8n::kInputWidth) *
                          static_cast<size_t>(mtkdemo::yolov8n::kInputHeight) * 3U) {
      use_int8_io = true;
    } else if (input_size == static_cast<size_t>(mtkdemo::yolov8n::kInputWidth) *
                                 static_cast<size_t>(mtkdemo::yolov8n::kInputHeight) * 3U *
                                 sizeof(float)) {
      use_int8_io = false;
    } else {
      throw std::runtime_error("Unsupported model input size for YOLOv8n demo: " +
                               std::to_string(input_size));
    }

    std::cout << "Model: " << options.model_path << "\n";
    std::cout << "Input dtype: " << (use_int8_io ? "int8" : "fp32") << "\n";
    if (options.skip_runtime) {
      mtkdemo::PrintNeuronRuntimeIoSummary(std::cout, runtime);
    }

    for (const std::string& image_path : options.image_paths) {
      mtkdemo::Image image = mtkdemo::LoadImage(image_path);
      mtkdemo::LetterboxInfo letterbox;

      std::vector<uint8_t> input_bytes;
      if (use_int8_io) {
        const std::vector<int8_t> input_tensor = mtkdemo::PreprocessToInt8RgbNchw(
            image, mtkdemo::yolov8n::kInputWidth, mtkdemo::yolov8n::kInputHeight,
            mtkdemo::yolov8n::kInt8InputQuant.scale, mtkdemo::yolov8n::kInt8InputQuant.zero_point,
            &letterbox);
        input_bytes = ToBytes(input_tensor);
      } else {
        const std::vector<float> input_tensor = mtkdemo::PreprocessToFloatRgbNchw(
            image, mtkdemo::yolov8n::kInputWidth, mtkdemo::yolov8n::kInputHeight, &letterbox);
        input_bytes = ToBytes(input_tensor);
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
        throw std::runtime_error("Model input size does not match current preprocessing output");
      }

      std::vector<std::vector<uint8_t>> outputs;
      runtime.Run({input_bytes}, &outputs);

      std::vector<mtkdemo::Detection> detections;
      if (use_int8_io) {
        std::vector<mtkdemo::YoloV8QuantParam> quant_params;
        quant_params.reserve(mtkdemo::yolov8n::kInt8OutputQuant.size());
        for (const auto& q : mtkdemo::yolov8n::kInt8OutputQuant) {
          quant_params.push_back(mtkdemo::YoloV8QuantParam{q.scale, q.zero_point});
        }
        detections = mtkdemo::DecodeYoloV8MultiOutputInt8(
            outputs, quant_params, letterbox, image.width, image.height,
            mtkdemo::yolov8n::kConfidenceThreshold, mtkdemo::yolov8n::kIouThreshold,
            mtkdemo::yolov8n::kClassCount, mtkdemo::yolov8n::kRegMax);
      } else {
        detections = mtkdemo::DecodeYoloV8MultiOutput(
            outputs, letterbox, image.width, image.height,
            mtkdemo::yolov8n::kConfidenceThreshold, mtkdemo::yolov8n::kIouThreshold,
            mtkdemo::yolov8n::kClassCount, mtkdemo::yolov8n::kRegMax);
      }

      mtkdemo::Image visualized = image;
      const auto& labels = mtkdemo::CocoLabels();
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
    std::cerr << "yolov8n_demo failed: " << ex.what() << "\n";
    return 1;
  }
}
