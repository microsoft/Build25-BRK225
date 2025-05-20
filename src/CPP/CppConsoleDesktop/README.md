# C++ Console Desktop Sample for ONNX Runtime

This sample demonstrates how to use ONNX Runtime in a C++ desktop application, focusing on:

1. Execution Provider (EP) discovery and configuration
2. Model compilation for optimized inference
3. Command-line options for flexible usage

## Command-line Usage

```
CppConsoleDesktop.exe [options] <image_path>
Options:
  --compile            Compile the model
  --download           Download required packages
  --model <path>       Path to input ONNX model (default: SqueezeNet.onnx in executable directory)
  --output <path>      Path for compiled output model
```

## Key Features

### 1. Execution Provider Configuration

The sample demonstrates how to discover available execution providers and configure them:

```cpp
#include <win_onnxruntime_cxx_api.h>

// Get all available EP devices from the environment
std::vector<Ort::ConstEpDevice> ep_devices = env.GetEpDevices();

// Configure and add each EP with appropriate options
for (const auto& device : ep_devices) {
    std::string ep_name = device.EpName();
    Ort::KeyValuePairs ep_options;
    
    if (ep_name == "QNNExecutionProvider") {
        ep_options.Add("htp_performance_mode", "high_performance");
        session_options.AppendExecutionProvider_V2(env, {device}, ep_options);
    }
    // ... other providers
}
```

### 2. Model Compilation

The sample shows how to compile an ONNX model for optimized execution:

```cpp
#include <win_onnxruntime_cxx_api.h>

// Get compile API
const OrtCompileApi* compileApi = ortApi.GetCompileApi();

// Create compilation options
OrtModelCompilationOptions* compileOptions = nullptr;
compileApi->CreateModelCompilationOptionsFromSessionOptions(env, sessionOptions, &compileOptions);

// Set input and output paths
compileApi->ModelCompilationOptions_SetInputModelPath(compileOptions, modelPath.c_str());
compileApi->ModelCompilationOptions_SetOutputModelPath(compileOptions, compiledModelPath.c_str());

// Compile the model
compileApi->CompileModel(env, compileOptions);
```

### 3. Execution Provider Selection Policy

The sample demonstrates how to set an EP selection policy to prefer specific hardware:

```cpp
#include <win_onnxruntime_cxx_api.h>

// Prefer NPU if available
Ort::SessionOptions sessionOptions;
sessionOptions.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_NPU);
```
