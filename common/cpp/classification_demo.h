#pragma once

#include <string>
#include <vector>

namespace mtkdemo {

struct ClassificationQuantParam {
  float scale = 1.0f;
  int zero_point = 0;
};

enum class ClassificationOutputKind {
  kLogits,
  kProbabilities,
};

struct ClassificationLabelMapping {
  // Effective class count used by Top-K and label mapping.
  int class_count = 1001;
  // Offset into the shared label table (e.g. 1000-class Keras models usually use 1 to skip background).
  int label_offset = 0;
  // Actual output tensor element count; <=0 means use class_count.
  int output_tensor_size = 0;
};

struct ClassificationDemoConfig {
  std::string demo_name;
  std::string default_int8_model_path;
  std::string default_fp32_model_path;
  std::vector<std::string> default_image_paths;
  std::string default_output_dir;
  int input_width = 224;
  int input_height = 224;
  int input_channels = 3;
  int top_k = 5;
  ClassificationQuantParam int8_input_quant;
  ClassificationQuantParam int8_output_quant;
  ClassificationOutputKind output_kind = ClassificationOutputKind::kLogits;
  ClassificationLabelMapping label_mapping;
};

int RunClassificationDemo(int argc, char** argv, const ClassificationDemoConfig& config);

}  // namespace mtkdemo
