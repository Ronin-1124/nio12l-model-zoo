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
#include "common/cpp/yolov8n-seg_postprocess.h"
#include "examples/yolov8n-seg/yolov8n-seg_config.h"

namespace fs = std::filesystem;

namespace {

struct Options {
  std::string model_path;
  std::vector<std::string> image_paths;
  std::string output_dir = mtkdemo::yolov8nseg::kDefaultOutputDir;
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
    options.model_path = options.use_fp32 ? mtkdemo::yolov8nseg::kDefaultFp32ModelPath
                                          : mtkdemo::yolov8nseg::kDefaultInt8ModelPath;
  }
  if (options.image_paths.empty()) {
    options.image_paths.assign(mtkdemo::yolov8nseg::kDefaultImagePaths.begin(),
                               mtkdemo::yolov8nseg::kDefaultImagePaths.end());
  }
  return options;
}

template <typename T>
std::vector<uint8_t> ToBytes(const std::vector<T>& values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

std::array<uint8_t, 3> ClassColor(int class_id) {
  const uint32_t seed = static_cast<uint32_t>(class_id * 2654435761U);
  return {static_cast<uint8_t>((seed >> 16) & 0xFF), static_cast<uint8_t>((seed >> 8) & 0xFF),
          static_cast<uint8_t>(seed & 0xFF)};
}

void OverlayMasks(mtkdemo::Image* image, const std::vector<mtkdemo::SegmentationInstance>& instances) {
  if (image == nullptr || image->channels < 3) return;
  const size_t plane = static_cast<size_t>(image->width) * static_cast<size_t>(image->height);
  for (const auto& inst : instances) {
    const auto color = ClassColor(inst.detection.class_id);
    for (size_t i = 0; i < plane && i < inst.mask.size(); ++i) {
      if (inst.mask[i] == 0) continue;
      const size_t idx = i * static_cast<size_t>(image->channels);
      image->data[idx + 0] =
          static_cast<uint8_t>(0.5f * image->data[idx + 0] + 0.5f * static_cast<float>(color[0]));
      image->data[idx + 1] =
          static_cast<uint8_t>(0.5f * image->data[idx + 1] + 0.5f * static_cast<float>(color[1]));
      image->data[idx + 2] =
          static_cast<uint8_t>(0.5f * image->data[idx + 2] + 0.5f * static_cast<float>(color[2]));
    }
  }
}

void WriteDetectionsJson(const fs::path& path,
                         const std::vector<mtkdemo::SegmentationInstance>& instances,
                         const std::vector<std::string>& labels) {
  std::ofstream stream(path);
  if (!stream) throw std::runtime_error("Failed to open detections json: " + path.string());
  stream << "[\n";
  for (size_t i = 0; i < instances.size(); ++i) {
    const auto& det = instances[i].detection;
    const std::string label =
        det.class_id >= 0 && static_cast<size_t>(det.class_id) < labels.size() ? labels[det.class_id]
                                                                                : "unknown";
    size_t mask_pixels = 0;
    for (uint8_t v : instances[i].mask) mask_pixels += (v != 0 ? 1U : 0U);
    stream << "  {\n"
           << "    \"class_id\": " << det.class_id << ",\n"
           << "    \"label\": \"" << label << "\",\n"
           << "    \"confidence\": " << std::fixed << std::setprecision(4) << det.confidence << ",\n"
           << "    \"x1\": " << det.x1 << ",\n"
           << "    \"y1\": " << det.y1 << ",\n"
           << "    \"x2\": " << det.x2 << ",\n"
           << "    \"y2\": " << det.y2 << ",\n"
           << "    \"mask_pixels\": " << mask_pixels << "\n"
           << "  }";
    if (i + 1 != instances.size()) stream << ",";
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
    if (runtime.InputCount() != 1 || runtime.OutputCount() != 10) {
      throw std::runtime_error("Current demo expects one-input ten-output YOLOv8n-seg model");
    }
    const size_t input_size = runtime.InputSize(0);
    if (input_size == static_cast<size_t>(mtkdemo::yolov8nseg::kInputWidth) *
                          static_cast<size_t>(mtkdemo::yolov8nseg::kInputHeight) * 3U) {
      use_int8_io = true;
    } else if (input_size == static_cast<size_t>(mtkdemo::yolov8nseg::kInputWidth) *
                                 static_cast<size_t>(mtkdemo::yolov8nseg::kInputHeight) * 3U *
                                 sizeof(float)) {
      use_int8_io = false;
    } else {
      throw std::runtime_error("Unsupported model input size for YOLOv8n-seg demo");
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
        const auto input = mtkdemo::PreprocessToInt8RgbNchw(
            image, mtkdemo::yolov8nseg::kInputWidth, mtkdemo::yolov8nseg::kInputHeight,
            mtkdemo::yolov8nseg::kInt8InputQuant.scale, mtkdemo::yolov8nseg::kInt8InputQuant.zero_point,
            &letterbox);
        input_bytes = ToBytes(input);
      } else {
        const auto input = mtkdemo::PreprocessToFloatRgbNchw(
            image, mtkdemo::yolov8nseg::kInputWidth, mtkdemo::yolov8nseg::kInputHeight, &letterbox);
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

      const std::vector<std::vector<uint8_t>> coeff_outputs = {outputs[0], outputs[1], outputs[2]};
      const std::vector<std::vector<uint8_t>> cls_outputs = {outputs[3], outputs[4], outputs[5]};
      const std::vector<std::vector<uint8_t>> bbox_outputs = {outputs[6], outputs[7], outputs[8]};
      const std::vector<uint8_t>& proto_output = outputs[9];

      std::vector<mtkdemo::SegmentationInstance> instances;
      if (use_int8_io) {
        std::vector<mtkdemo::YoloV8SegQuantParam> coeff_q;
        std::vector<mtkdemo::YoloV8SegQuantParam> cls_q;
        std::vector<mtkdemo::YoloV8SegQuantParam> bbox_q;
        for (const auto& q : mtkdemo::yolov8nseg::kInt8CoeffQuant)
          coeff_q.push_back({q.scale, q.zero_point});
        for (const auto& q : mtkdemo::yolov8nseg::kInt8ClsQuant)
          cls_q.push_back({q.scale, q.zero_point});
        for (const auto& q : mtkdemo::yolov8nseg::kInt8BboxQuant)
          bbox_q.push_back({q.scale, q.zero_point});
        instances = mtkdemo::DecodeYoloV8SegMultiOutputInt8(
            coeff_outputs, coeff_q, cls_outputs, cls_q, bbox_outputs, bbox_q, proto_output,
            {mtkdemo::yolov8nseg::kInt8ProtoQuant.scale, mtkdemo::yolov8nseg::kInt8ProtoQuant.zero_point},
            letterbox, image.width, image.height, mtkdemo::yolov8nseg::kConfidenceThreshold,
            mtkdemo::yolov8nseg::kIouThreshold, mtkdemo::yolov8nseg::kMaskThreshold,
            mtkdemo::yolov8nseg::kClassCount, mtkdemo::yolov8nseg::kRegMax,
            mtkdemo::yolov8nseg::kInputWidth, mtkdemo::yolov8nseg::kProtoHeight,
            mtkdemo::yolov8nseg::kProtoWidth);
      } else {
        instances = mtkdemo::DecodeYoloV8SegMultiOutput(
            coeff_outputs, cls_outputs, bbox_outputs, proto_output, letterbox, image.width, image.height,
            mtkdemo::yolov8nseg::kConfidenceThreshold, mtkdemo::yolov8nseg::kIouThreshold,
            mtkdemo::yolov8nseg::kMaskThreshold, mtkdemo::yolov8nseg::kClassCount,
            mtkdemo::yolov8nseg::kRegMax, mtkdemo::yolov8nseg::kInputWidth,
            mtkdemo::yolov8nseg::kProtoHeight, mtkdemo::yolov8nseg::kProtoWidth);
      }

      std::vector<mtkdemo::Detection> dets;
      dets.reserve(instances.size());
      for (const auto& inst : instances) dets.push_back(inst.detection);

      mtkdemo::Image vis = image;
      OverlayMasks(&vis, instances);
      mtkdemo::DrawDetections(&vis, dets, mtkdemo::CocoLabels());

      const fs::path vis_path = vis_dir / (image_stem.string() + "_seg.png");
      mtkdemo::SaveImage(vis_path.string(), vis);
      const fs::path json_path = det_dir / (image_stem.string() + ".json");
      WriteDetectionsJson(json_path, instances, mtkdemo::CocoLabels());
      std::cout << "Visualized result: " << vis_path << "\n";
      std::cout << "Detections json: " << json_path << "\n";
      std::cout << "Instances: " << instances.size() << "\n";
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "yolov8n-seg_demo failed: " << ex.what() << "\n";
    return 1;
  }
}
