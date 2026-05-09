#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace mtkdemo {

class NeuronRuntimeRunner {
 public:
  NeuronRuntimeRunner() = default;
  ~NeuronRuntimeRunner();

  NeuronRuntimeRunner(const NeuronRuntimeRunner&) = delete;
  NeuronRuntimeRunner& operator=(const NeuronRuntimeRunner&) = delete;

  void Load(const std::string& model_path, size_t threads = 1, size_t backlog = 1,
            const std::string& options = "");
  size_t InputCount() const;
  size_t OutputCount() const;
  size_t InputSize(size_t index) const;
  size_t OutputSize(size_t index) const;
  void Run(const std::vector<std::vector<uint8_t>>& inputs,
           std::vector<std::vector<uint8_t>>* outputs) const;

 private:
  void* runtime_ = nullptr;
  std::vector<size_t> input_sizes_;
  std::vector<size_t> output_sizes_;
};

// Prints input/output buffer byte sizes (from NeuronRuntimeV2_get*Size). Requires a loaded runner.
void PrintNeuronRuntimeIoSummary(std::ostream& os, const NeuronRuntimeRunner& runtime);

}  // namespace mtkdemo
