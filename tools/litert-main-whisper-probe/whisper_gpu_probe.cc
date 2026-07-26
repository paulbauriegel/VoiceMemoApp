#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tflite/delegates/gpu/delegate.h"
#include "tflite/delegates/gpu/delegate_options.h"
#include "tflite/kernels/kernel_util.h"
#include "tflite/kernels/register.h"
#include "tflite/model_builder.h"
#include "tflite/schema/schema_generated.h"

namespace {

constexpr int kEncoderFrames = 1500;
constexpr int kEncoderWidth = 1280;
constexpr int kEncoderStateFloats = kEncoderFrames * kEncoderWidth;
constexpr int kTargetLength = 128;
constexpr int kMaskFloats = kTargetLength * kTargetLength;
constexpr int kBoundaryOutputFloats = kTargetLength * kEncoderWidth;
constexpr float kMaskedLogit = -1.0e9f;

constexpr int kEosToken = 50257;
constexpr int kDecoderStartToken = 50258;
constexpr int kRussianLanguageToken = 50263;
constexpr int kTranscribeToken = 50360;
constexpr int kNoTimestampsToken = 50364;

using Clock = std::chrono::steady_clock;

struct InputTensors {
  TfLiteTensor* encoder_state = nullptr;
  TfLiteTensor* tokens = nullptr;
  TfLiteTensor* mask = nullptr;
};

struct StepResult {
  std::vector<float> row;
  double milliseconds = 0.0;
};

struct Comparison {
  int exact_elements = 0;
  int cpu_argmax = -1;
  int gpu_argmax = -1;
  double max_absolute_error = 0.0;
  double mean_absolute_error = 0.0;
  double root_mean_square_error = 0.0;
  double cosine_similarity = 0.0;
};

double ElapsedMilliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

bool ParseNodeIndex(const char* value, int* parsed) {
  char* end = nullptr;
  const long result = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || result < 0 ||
      result > std::numeric_limits<int>::max()) {
    return false;
  }
  *parsed = static_cast<int>(result);
  return true;
}

void PrintOperationPlan(const tflite::Interpreter& interpreter) {
  std::map<int, int> counts;
  for (int node_index : interpreter.execution_plan()) {
    const auto* node_and_registration =
        interpreter.node_and_registration(node_index);
    if (node_and_registration == nullptr) {
      continue;
    }
    const int builtin_code = node_and_registration->second.builtin_code;
    const auto builtin =
        static_cast<tflite::BuiltinOperator>(builtin_code);
    ++counts[builtin_code];
    std::cout << "op node=" << node_index
              << " builtin=" << builtin_code
              << " name=" << tflite::EnumNameBuiltinOperator(builtin)
              << std::endl;
  }
  for (const auto& [builtin_code, count] : counts) {
    const auto builtin =
        static_cast<tflite::BuiltinOperator>(builtin_code);
    std::cout << "op_count builtin=" << builtin_code
              << " name=" << tflite::EnumNameBuiltinOperator(builtin)
              << " count=" << count << std::endl;
  }
}

bool FindInputTensors(tflite::Interpreter* interpreter, InputTensors* inputs) {
  for (int input_index : interpreter->inputs()) {
    TfLiteTensor* tensor = interpreter->tensor(input_index);
    const int elements = tflite::NumElements(tensor);
    if (tensor->type == kTfLiteFloat32 &&
        elements == kEncoderStateFloats) {
      inputs->encoder_state = tensor;
    } else if (tensor->type == kTfLiteInt32 &&
               elements == kTargetLength) {
      inputs->tokens = tensor;
    } else if (tensor->type == kTfLiteFloat32 &&
               elements == kMaskFloats) {
      inputs->mask = tensor;
    }
  }
  return inputs->encoder_state != nullptr && inputs->tokens != nullptr &&
         inputs->mask != nullptr;
}

void FillInputs(tflite::Interpreter* interpreter,
                const std::vector<int>& prefix) {
  InputTensors inputs;
  if (!FindInputTensors(interpreter, &inputs)) {
    std::cerr << "Could not resolve the Whisper decoder inputs." << std::endl;
    std::exit(3);
  }

  for (int index = 0; index < kEncoderStateFloats; ++index) {
    const int frame = index / kEncoderWidth;
    const int channel = index % kEncoderWidth;
    const double slow = std::sin(frame * 0.013 + channel * 0.007);
    const double cross = std::cos(frame * 0.031 - channel * 0.011);
    inputs.encoder_state->data.f[index] =
        static_cast<float>(0.17 * slow + 0.11 * cross);
  }

  std::vector<int32_t> prompt = {
      kDecoderStartToken,
      kRussianLanguageToken,
      kTranscribeToken,
      kNoTimestampsToken,
  };
  prompt.insert(prompt.end(), prefix.begin(), prefix.end());
  for (int index = 0; index < kTargetLength; ++index) {
    inputs.tokens->data.i32[index] =
        index < prompt.size() ? prompt[index] : kEosToken;
  }

  for (int row = 0; row < kTargetLength; ++row) {
    for (int column = 0; column < kTargetLength; ++column) {
      inputs.mask->data.f[row * kTargetLength + column] =
          column <= row ? 0.0f : kMaskedLogit;
    }
  }
}

StepResult InvokeStep(tflite::Interpreter* interpreter,
                      const std::vector<int>& prefix) {
  FillInputs(interpreter, prefix);
  const auto start = Clock::now();
  if (interpreter->Invoke() != kTfLiteOk) {
    std::cerr << "Whisper decoder invocation failed." << std::endl;
    std::exit(4);
  }

  if (interpreter->outputs().size() != 1) {
    std::cerr << "Expected one Whisper decoder output." << std::endl;
    std::exit(5);
  }
  const TfLiteTensor* output =
      interpreter->tensor(interpreter->outputs().front());
  if (output->type != kTfLiteFloat32 ||
      tflite::NumElements(output) != kBoundaryOutputFloats) {
    std::cerr << "Unexpected Whisper decoder output contract." << std::endl;
    std::exit(6);
  }

  const int active_row = 3 + prefix.size();
  const float* row_start = output->data.f + active_row * kEncoderWidth;
  return {
      std::vector<float>(row_start, row_start + kEncoderWidth),
      ElapsedMilliseconds(start),
  };
}

int Argmax(const std::vector<float>& values) {
  return std::distance(
      values.begin(),
      std::max_element(values.begin(), values.end()));
}

Comparison Compare(const std::vector<float>& cpu,
                   const std::vector<float>& gpu) {
  Comparison result;
  double absolute_error_sum = 0.0;
  double squared_error_sum = 0.0;
  double cpu_squared_sum = 0.0;
  double gpu_squared_sum = 0.0;
  double dot = 0.0;
  for (int index = 0; index < cpu.size(); ++index) {
    if (std::memcmp(&cpu[index], &gpu[index], sizeof(float)) == 0) {
      ++result.exact_elements;
    }
    const double expected = cpu[index];
    const double actual = gpu[index];
    const double difference = actual - expected;
    const double absolute_error = std::abs(difference);
    result.max_absolute_error =
        std::max(result.max_absolute_error, absolute_error);
    absolute_error_sum += absolute_error;
    squared_error_sum += difference * difference;
    cpu_squared_sum += expected * expected;
    gpu_squared_sum += actual * actual;
    dot += expected * actual;
  }
  const double count = cpu.size();
  result.cpu_argmax = Argmax(cpu);
  result.gpu_argmax = Argmax(gpu);
  result.mean_absolute_error = absolute_error_sum / count;
  result.root_mean_square_error = std::sqrt(squared_error_sum / count);
  result.cosine_similarity =
      dot / std::sqrt(cpu_squared_sum * gpu_squared_sum);
  return result;
}

std::unique_ptr<tflite::Interpreter> CreateInterpreter(
    const tflite::FlatBufferModel& model,
    const tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates&
        resolver,
    int threads) {
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder builder(model, resolver);
  if (builder(&interpreter) != kTfLiteOk || interpreter == nullptr) {
    return nullptr;
  }
  if (interpreter->SetNumThreads(threads) != kTfLiteOk) {
    return nullptr;
  }
  return interpreter;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 && argc != 3 && argc != 5) {
    std::cerr
        << "Usage: whisper_gpu_probe CANDIDATE.tflite "
           "[CPU_REFERENCE.tflite [FIRST_GPU_NODE LAST_GPU_NODE]]"
        << std::endl;
    return 1;
  }

  int first_gpu_node = 0;
  int last_gpu_node = std::numeric_limits<int>::max();
  if (argc == 5 &&
      (!ParseNodeIndex(argv[3], &first_gpu_node) ||
       !ParseNodeIndex(argv[4], &last_gpu_node) ||
       first_gpu_node > last_gpu_node)) {
    std::cerr << "GPU node range must contain two ordered non-negative "
                 "integers."
              << std::endl;
    return 1;
  }

  auto model = tflite::FlatBufferModel::BuildFromFile(argv[1]);
  if (model == nullptr) {
    std::cerr << "Could not load model: " << argv[1] << std::endl;
    return 1;
  }
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;

  auto cpu = CreateInterpreter(*model, resolver, 3);
  if (cpu == nullptr || cpu->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Could not prepare CPU reference." << std::endl;
    return 1;
  }
  const char* print_operations = std::getenv("VOICEMEMO_PRINT_OPS");
  if (print_operations != nullptr &&
      std::strcmp(print_operations, "1") == 0) {
    PrintOperationPlan(*cpu);
  }

  const std::vector<std::vector<int>> prefixes = {
      {},
      {407},
      {407, 5616},
  };
  std::vector<StepResult> candidate_cpu_steps;
  for (const auto& prefix : prefixes) {
    candidate_cpu_steps.push_back(InvokeStep(cpu.get(), prefix));
  }

  std::unique_ptr<tflite::FlatBufferModel> reference_model;
  std::unique_ptr<tflite::Interpreter> reference_cpu;
  std::vector<StepResult> reference_cpu_steps;
  if (argc == 3) {
    reference_model = tflite::FlatBufferModel::BuildFromFile(argv[2]);
    if (reference_model == nullptr) {
      std::cerr << "Could not load CPU reference model: " << argv[2]
                << std::endl;
      return 1;
    }
    reference_cpu = CreateInterpreter(*reference_model, resolver, 3);
    if (reference_cpu == nullptr ||
        reference_cpu->AllocateTensors() != kTfLiteOk) {
      std::cerr << "Could not prepare the original-model CPU reference."
                << std::endl;
      return 1;
    }
    for (const auto& prefix : prefixes) {
      reference_cpu_steps.push_back(InvokeStep(reference_cpu.get(), prefix));
    }
  }

  auto gpu = CreateInterpreter(*model, resolver, 1);
  if (gpu == nullptr) {
    std::cerr << "Could not create GPU candidate interpreter." << std::endl;
    return 1;
  }
  TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
  options.is_precision_loss_allowed = 0;
  options.inference_preference =
      TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
  options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;
  options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_CL_ONLY;
  options.max_delegated_partitions = 1000;
#ifdef TFLITE_DEBUG_DELEGATE
  options.first_delegate_node_index = first_gpu_node;
  options.last_delegate_node_index = last_gpu_node;
#else
  if (argc == 5) {
    std::cerr << "This probe was not built with TFLITE_DEBUG_DELEGATE."
              << std::endl;
    return 1;
  }
#endif
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteGpuDelegateV2Delete)> delegate(
      TfLiteGpuDelegateV2Create(&options), TfLiteGpuDelegateV2Delete);
  if (delegate == nullptr ||
      gpu->ModifyGraphWithDelegate(delegate.get()) != kTfLiteOk) {
    std::cerr << "Could not apply current-main OpenCL delegate." << std::endl;
    return 2;
  }
  if (gpu->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Could not allocate GPU candidate tensors." << std::endl;
    return 2;
  }

  std::cout << std::setprecision(12);
  std::cout << "candidate_cpu_execution_plan="
            << cpu->execution_plan().size()
            << std::endl;
  if (reference_cpu != nullptr) {
    std::cout << "reference_cpu_execution_plan="
              << reference_cpu->execution_plan().size() << std::endl;
  }
  std::cout << "gpu_execution_plan=" << gpu->execution_plan().size()
            << std::endl;
  std::cout << "gpu_first_node=" << first_gpu_node
            << " gpu_last_node=" << last_gpu_node << std::endl;
  const char* excluded_builtins =
      std::getenv("VOICEMEMO_GPU_EXCLUDED_BUILTINS");
  std::cout << "gpu_excluded_builtins="
            << (excluded_builtins == nullptr ? "" : excluded_builtins)
            << std::endl;
  for (int step = 0; step < prefixes.size(); ++step) {
    const StepResult gpu_step = InvokeStep(gpu.get(), prefixes[step]);
    const Comparison gpu_comparison =
        Compare(candidate_cpu_steps[step].row, gpu_step.row);
    std::cout << "step=" << step
              << " candidate_cpu_ms="
              << candidate_cpu_steps[step].milliseconds;
    if (reference_cpu != nullptr) {
      const Comparison cpu_comparison =
          Compare(
              reference_cpu_steps[step].row,
              candidate_cpu_steps[step].row);
      std::cout << " reference_cpu_ms="
                << reference_cpu_steps[step].milliseconds
                << " cpu_parity_exact_elements="
                << cpu_comparison.exact_elements
                << " cpu_parity_max_abs_error="
                << cpu_comparison.max_absolute_error
                << " cpu_parity_mean_abs_error="
                << cpu_comparison.mean_absolute_error
                << " cpu_parity_rmse="
                << cpu_comparison.root_mean_square_error
                << " cpu_parity_cosine="
                << cpu_comparison.cosine_similarity;
    }
    std::cout
              << " gpu_ms=" << gpu_step.milliseconds
              << " gpu_exact_elements=" << gpu_comparison.exact_elements
              << " element_count=" << kEncoderWidth
              << " candidate_cpu_argmax="
              << gpu_comparison.cpu_argmax
              << " gpu_argmax=" << gpu_comparison.gpu_argmax
              << " gpu_argmax_match="
              << (gpu_comparison.cpu_argmax == gpu_comparison.gpu_argmax
                      ? "true"
                      : "false")
              << " gpu_max_abs_error="
              << gpu_comparison.max_absolute_error
              << " gpu_mean_abs_error="
              << gpu_comparison.mean_absolute_error
              << " gpu_rmse="
              << gpu_comparison.root_mean_square_error
              << " gpu_cosine="
              << gpu_comparison.cosine_similarity << std::endl;
  }
  return 0;
}
