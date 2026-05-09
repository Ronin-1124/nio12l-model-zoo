#include "common/cpp/classification_demo.h"

#include <algorithm>
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

#include "common/cpp/image_io.h"
#include "common/cpp/imagenet_labels.h"
#include "common/cpp/neuron_runtime.h"

namespace fs = std::filesystem;

namespace mtkdemo {
namespace {

struct Options {
  std::string model_path;
  std::vector<std::string> image_paths;
  std::string output_dir;
  size_t threads = 1;
  size_t backlog = 1;
  bool skip_runtime = false;
  bool use_fp32 = false;
  bool model_overridden = false;
  std::string runtime_options;
};

struct Classification {
  int class_id = -1;
  float score = 0.0f;
};

void PrintUsage(const char* argv0) {
  std::cout << "Usage: " << argv0
            << " [--fp32] [--model path] [--image path ...] [--output-dir dir]"
            << " [--threads n] [--backlog n] [--runtime-options opts] [--skip-runtime]\n";
}

Options ParseArgs(int argc, char** argv, const ClassificationDemoConfig& config) {
  Options options;
  options.output_dir = config.default_output_dir;
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
    options.model_path =
        options.use_fp32 ? config.default_fp32_model_path : config.default_int8_model_path;
  }
  if (options.image_paths.empty()) {
    options.image_paths = config.default_image_paths;
  }
  return options;
}

template <typename T>
std::vector<uint8_t> ToBytes(const std::vector<T>& values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

inline uint8_t PixelAt(const Image& image, int x, int y, int channel) {
  const size_t index =
      (static_cast<size_t>(y) * static_cast<size_t>(image.width) + static_cast<size_t>(x)) * 3U +
      static_cast<size_t>(channel);
  return image.data[index];
}

std::vector<float> ResizeToFloatRgbNhwc(const Image& image, int target_width, int target_height) {
  if (image.channels != 3) throw std::runtime_error("Only 3-channel RGB input is supported");
  std::vector<float> output(static_cast<size_t>(target_width) * static_cast<size_t>(target_height) * 3U);
  const float scale_x = static_cast<float>(image.width) / static_cast<float>(target_width);
  const float scale_y = static_cast<float>(image.height) / static_cast<float>(target_height);

  for (int y = 0; y < target_height; ++y) {
    const float src_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
    const int y0 = std::clamp(static_cast<int>(std::floor(src_y)), 0, image.height - 1);
    const int y1 = std::clamp(y0 + 1, 0, image.height - 1);
    const float ly = src_y - std::floor(src_y);
    for (int x = 0; x < target_width; ++x) {
      const float src_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
      const int x0 = std::clamp(static_cast<int>(std::floor(src_x)), 0, image.width - 1);
      const int x1 = std::clamp(x0 + 1, 0, image.width - 1);
      const float lx = src_x - std::floor(src_x);
      const size_t dst_index =
          (static_cast<size_t>(y) * static_cast<size_t>(target_width) + static_cast<size_t>(x)) * 3U;
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
  return output;
}

std::vector<int8_t> ResizeToInt8RgbNhwc(const Image& image, int target_width, int target_height,
                                        ClassificationQuantParam quant) {
  if (quant.scale <= 0.0f) throw std::runtime_error("INT8 preprocess scale must be positive");
  const std::vector<float> input = ResizeToFloatRgbNhwc(image, target_width, target_height);
  std::vector<int8_t> quantized(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    const float q = std::round(input[i] / quant.scale) + static_cast<float>(quant.zero_point);
    quantized[i] = static_cast<int8_t>(std::clamp(static_cast<int>(q), -128, 127));
  }
  return quantized;
}

std::vector<float> OutputToFloat(const std::vector<uint8_t>& output, bool int8_output,
                                 ClassificationQuantParam quant, const std::string& demo_name) {
  if (int8_output) {
    std::vector<float> values(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
      const int8_t q = reinterpret_cast<const int8_t*>(output.data())[i];
      values[i] = (static_cast<float>(q) - static_cast<float>(quant.zero_point)) * quant.scale;
    }
    return values;
  }
  if (output.size() % sizeof(float) != 0) {
    throw std::runtime_error(demo_name + " output is not fp32");
  }
  const size_t count = output.size() / sizeof(float);
  std::vector<float> values(count);
  std::memcpy(values.data(), output.data(), output.size());
  return values;
}

std::vector<float> Softmax(const std::vector<float>& logits) {
  if (logits.empty()) return {};
  const float max_logit = *std::max_element(logits.begin(), logits.end());
  std::vector<float> probs(logits.size());
  float sum = 0.0f;
  for (size_t i = 0; i < logits.size(); ++i) {
    probs[i] = std::exp(logits[i] - max_logit);
    sum += probs[i];
  }
  if (sum <= 0.0f) return probs;
  for (float& p : probs) p /= sum;
  return probs;
}

std::vector<Classification> TopK(const std::vector<float>& scores, int k) {
  std::vector<int> indices(scores.size());
  for (size_t i = 0; i < indices.size(); ++i) indices[i] = static_cast<int>(i);
  const size_t keep = std::min(static_cast<size_t>(std::max(k, 0)), indices.size());
  std::partial_sort(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(keep), indices.end(),
                    [&scores](int lhs, int rhs) {
                      return scores[static_cast<size_t>(lhs)] > scores[static_cast<size_t>(rhs)];
                    });
  std::vector<Classification> results;
  results.reserve(keep);
  for (size_t i = 0; i < keep; ++i) {
    const int idx = indices[i];
    results.push_back(Classification{idx, scores[static_cast<size_t>(idx)]});
  }
  return results;
}

void WriteClassificationsJson(const fs::path& path, const std::vector<Classification>& results,
                              const std::vector<std::string>& labels) {
  std::ofstream stream(path);
  if (!stream) throw std::runtime_error("Failed to open classification json: " + path.string());
  stream << "[\n";
  for (size_t i = 0; i < results.size(); ++i) {
    const auto& result = results[i];
    const std::string label = result.class_id >= 0 && static_cast<size_t>(result.class_id) < labels.size()
                                  ? labels[static_cast<size_t>(result.class_id)]
                                  : "unknown";
    stream << "  {\n"
           << "    \"class_id\": " << result.class_id << ",\n"
           << "    \"label\": \"" << label << "\",\n"
           << "    \"score\": " << std::fixed << std::setprecision(6) << result.score << "\n"
           << "  }";
    if (i + 1 != results.size()) stream << ",";
    stream << "\n";
  }
  stream << "]\n";
}

}  // namespace

int RunClassificationDemo(int argc, char** argv, const ClassificationDemoConfig& config) {
  try {
    const Options options = ParseArgs(argc, argv, config);
    const fs::path output_dir = options.output_dir;
    const fs::path cls_dir = output_dir / "classifications";
    fs::create_directories(cls_dir);

    bool use_int8_io = !options.use_fp32;
    NeuronRuntimeRunner runtime;
    runtime.Load(options.model_path, options.threads, options.backlog, options.runtime_options);
    if (runtime.InputCount() != 1 || runtime.OutputCount() != 1) {
      throw std::runtime_error("Current demo expects one-input one-output " + config.demo_name + " model");
    }

    const size_t expected_int8_input = static_cast<size_t>(config.input_width) *
                                       static_cast<size_t>(config.input_height) *
                                       static_cast<size_t>(config.input_channels);
    const size_t expected_fp32_input = expected_int8_input * sizeof(float);
    if (runtime.InputSize(0) == expected_int8_input) {
      use_int8_io = true;
    } else if (runtime.InputSize(0) == expected_fp32_input) {
      use_int8_io = false;
    } else {
      throw std::runtime_error("Unsupported model input size for " + config.demo_name + " demo");
    }
    const int class_count = config.label_mapping.class_count;
    const int label_offset = config.label_mapping.label_offset;
    const int output_tensor_size_cfg = config.label_mapping.output_tensor_size;
    if (class_count <= 0) {
      throw std::runtime_error(config.demo_name + ": class_count must be positive");
    }
    const size_t output_tensor_size = output_tensor_size_cfg > 0
                                          ? static_cast<size_t>(output_tensor_size_cfg)
                                          : static_cast<size_t>(class_count);
    const bool int8_output = runtime.OutputSize(0) == output_tensor_size;

    std::cout << "Model: " << options.model_path << "\n";
    std::cout << "Input dtype: " << (use_int8_io ? "int8" : "fp32") << "\n";
    std::cout << "Output dtype: " << (int8_output ? "int8" : "fp32") << "\n";
    if (options.skip_runtime) PrintNeuronRuntimeIoSummary(std::cout, runtime);

    const auto& full_labels = ImagenetLabels();
    if (label_offset < 0 ||
        static_cast<size_t>(label_offset) + static_cast<size_t>(class_count) >
            full_labels.size()) {
      throw std::runtime_error(config.demo_name +
                               ": label_offset/class_count exceed shared ImageNet labels size");
    }
    const std::vector<std::string> labels(
        full_labels.begin() + label_offset, full_labels.begin() + label_offset + class_count);
    for (const std::string& image_path : options.image_paths) {
      const Image image = LoadImage(image_path);
      std::vector<uint8_t> input_bytes;
      if (use_int8_io) {
        const auto input =
            ResizeToInt8RgbNhwc(image, config.input_width, config.input_height, config.int8_input_quant);
        input_bytes = ToBytes(input);
      } else {
        const auto input = ResizeToFloatRgbNhwc(image, config.input_width, config.input_height);
        input_bytes = ToBytes(input);
      }

      const fs::path image_stem = fs::path(image_path).stem();
      std::cout << "Image: " << image_path << "\n";
      if (runtime.InputSize(0) != input_bytes.size()) {
        throw std::runtime_error("Model input size does not match preprocessing output");
      }
      if (options.skip_runtime) {
        std::cout << "Preprocess output: " << input_bytes.size()
                  << " bytes (matches model input[0])\n";
        std::cout << "Skipping NeuronRuntimeV2_run (--skip-runtime).\n";
        continue;
      }

      std::vector<std::vector<uint8_t>> outputs;
      runtime.Run({input_bytes}, &outputs);
      if (outputs.size() != 1) throw std::runtime_error(config.demo_name + " runtime returned unexpected outputs");
      if (runtime.OutputSize(0) != outputs[0].size()) {
        throw std::runtime_error("Model output size does not match runtime output buffer");
      }
      std::vector<float> raw =
          OutputToFloat(outputs[0], int8_output, config.int8_output_quant, config.demo_name);
      if (raw.size() != output_tensor_size) {
        throw std::runtime_error(config.demo_name + " output tensor size mismatch");
      }
      // Tensor length may exceed effective class count (e.g. ResNet50: 1001 outputs with 1000 valid classes); truncate to class_count.
      raw.resize(static_cast<size_t>(class_count));
      const std::vector<float> probs =
          config.output_kind == ClassificationOutputKind::kProbabilities ? std::move(raw)
                                                                         : Softmax(raw);
      const std::vector<Classification> topk = TopK(probs, config.top_k);

      const fs::path json_path = cls_dir / (image_stem.string() + ".json");
      WriteClassificationsJson(json_path, topk, labels);
      std::cout << "Classifications json: " << json_path << "\n";
      if (!topk.empty()) {
        const int top_id = topk.front().class_id;
        const std::string top_label = top_id >= 0 && static_cast<size_t>(top_id) < labels.size()
                                          ? labels[static_cast<size_t>(top_id)]
                                          : "unknown";
        std::cout << "Top-1: " << top_id << " " << top_label << " ("
                  << std::fixed << std::setprecision(4) << topk.front().score << ")\n";
      }
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << config.demo_name << "_demo failed: " << ex.what() << "\n";
    return 1;
  }
}

}  // namespace mtkdemo
