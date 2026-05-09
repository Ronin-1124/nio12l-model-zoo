#include "common/cpp/neuron_runtime.h"

#include <cstring>
#include <ostream>
#include <stdexcept>

#include "neuron/api/RuntimeV2.h"

namespace mtkdemo {
namespace {

void CheckStatus(int status, const char* message) {
  if (status != 0) {
    throw std::runtime_error(std::string(message) + ", status=" + std::to_string(status));
  }
}

}  // namespace

NeuronRuntimeRunner::~NeuronRuntimeRunner() {
  if (runtime_ != nullptr) {
    NeuronRuntimeV2_release(runtime_);
    runtime_ = nullptr;
  }
}

void NeuronRuntimeRunner::Load(const std::string& model_path, size_t threads, size_t backlog,
                                 const std::string& options) {
  if (runtime_ != nullptr) {
    NeuronRuntimeV2_release(runtime_);
    runtime_ = nullptr;
  }

  if (options.empty()) {
    CheckStatus(NeuronRuntimeV2_create(model_path.c_str(), threads, &runtime_, backlog),
                "NeuronRuntimeV2_create failed");
  } else {
    CheckStatus(NeuronRuntimeV2_create_with_options(model_path.c_str(), threads, &runtime_, backlog,
                                                   options.c_str()),
                "NeuronRuntimeV2_create_with_options failed");
  }

  size_t input_count = 0;
  size_t output_count = 0;
  CheckStatus(NeuronRuntimeV2_getInputNumber(runtime_, &input_count),
              "NeuronRuntimeV2_getInputNumber failed");
  CheckStatus(NeuronRuntimeV2_getOutputNumber(runtime_, &output_count),
              "NeuronRuntimeV2_getOutputNumber failed");

  input_sizes_.assign(input_count, 0);
  output_sizes_.assign(output_count, 0);

  for (size_t i = 0; i < input_count; ++i) {
    CheckStatus(NeuronRuntimeV2_getInputSize(runtime_, i, &input_sizes_[i]),
                "NeuronRuntimeV2_getInputSize failed");
  }
  for (size_t i = 0; i < output_count; ++i) {
    CheckStatus(NeuronRuntimeV2_getOutputSize(runtime_, i, &output_sizes_[i]),
                "NeuronRuntimeV2_getOutputSize failed");
  }
}

size_t NeuronRuntimeRunner::InputCount() const { return input_sizes_.size(); }

size_t NeuronRuntimeRunner::OutputCount() const { return output_sizes_.size(); }

size_t NeuronRuntimeRunner::InputSize(size_t index) const { return input_sizes_.at(index); }

size_t NeuronRuntimeRunner::OutputSize(size_t index) const { return output_sizes_.at(index); }

void NeuronRuntimeRunner::Run(const std::vector<std::vector<uint8_t>>& inputs,
                              std::vector<std::vector<uint8_t>>* outputs) const {
  if (runtime_ == nullptr) {
    throw std::runtime_error("Runtime is not loaded");
  }
  if (outputs == nullptr) {
    throw std::runtime_error("Output buffer container must not be null");
  }
  if (inputs.size() != input_sizes_.size()) {
    throw std::runtime_error("Input buffer count does not match model input count");
  }

  outputs->assign(output_sizes_.size(), {});
  for (size_t i = 0; i < output_sizes_.size(); ++i) {
    (*outputs)[i].resize(output_sizes_[i]);
  }

  std::vector<IOBuffer> input_buffers;
  std::vector<IOBuffer> output_buffers;
  input_buffers.reserve(input_sizes_.size());
  output_buffers.reserve(output_sizes_.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].size() != input_sizes_[i]) {
      throw std::runtime_error("Input buffer size does not match model input size");
    }
    input_buffers.emplace_back(const_cast<uint8_t*>(inputs[i].data()), inputs[i].size(), -1, 0);
    std::memset(&input_buffers.back().reserved1_should_be_init_zero, 0,
                sizeof(input_buffers.back().reserved1_should_be_init_zero) +
                    sizeof(input_buffers.back().reserved2_should_be_init_zero) +
                    sizeof(input_buffers.back().reserved3_should_be_init_zero));
  }

  for (size_t i = 0; i < output_sizes_.size(); ++i) {
    output_buffers.emplace_back((*outputs)[i].data(), (*outputs)[i].size(), -1, 0);
    std::memset(&output_buffers.back().reserved1_should_be_init_zero, 0,
                sizeof(output_buffers.back().reserved1_should_be_init_zero) +
                    sizeof(output_buffers.back().reserved2_should_be_init_zero) +
                    sizeof(output_buffers.back().reserved3_should_be_init_zero));
  }

  SyncInferenceRequest request;
  request.inputs = input_buffers.data();
  request.outputs = output_buffers.data();

  CheckStatus(NeuronRuntimeV2_run(runtime_, request), "NeuronRuntimeV2_run failed");
}

void PrintNeuronRuntimeIoSummary(std::ostream& os, const NeuronRuntimeRunner& runtime) {
  os << "Neuron model I/O (byte sizes):\n";
  for (size_t i = 0; i < runtime.InputCount(); ++i) {
    os << "  input[" << i << "]: " << runtime.InputSize(i) << " bytes\n";
  }
  for (size_t i = 0; i < runtime.OutputCount(); ++i) {
    os << "  output[" << i << "]: " << runtime.OutputSize(i) << " bytes\n";
  }
}

}  // namespace mtkdemo
