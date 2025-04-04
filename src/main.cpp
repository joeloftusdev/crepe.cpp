#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>
#include <fstream>

int main() {
    try {
        const Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "CREPE");

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

        const std::string model_path = "../external/model.ort";

        // Check if the model file exists
        if (std::ifstream model_file(model_path); !model_file.good()) {
            std::cerr << "Model file not found: " << model_path << std::endl;
            return 1;
        }

        Ort::Session onnx_session(env, model_path.c_str(), session_options);

        // expected dimensions for the input tensor
        const std::vector<int64_t> input_dims = {1, 1024};
        std::vector<float> input_data(1024);  // populate this with actual audio waveform data

        // input tensor
        const Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(),
            input_dims.data(), input_dims.size());

        // input and output names from the model
        Ort::AllocatorWithDefaultOptions allocator;

        // input names
        size_t num_input_nodes = onnx_session.GetInputCount();
        std::vector<std::string> input_name_strings;
        std::vector<const char*> input_names_ptr;

        for (size_t i = 0; i < num_input_nodes; i++) {
            Ort::AllocatedStringPtr input_name_ptr = onnx_session.GetInputNameAllocated(i, allocator);
            const char* input_name = input_name_ptr.get();
            std::string input_name_str(input_name);
            input_name_strings.push_back(input_name_str);
            std::cout << "Input " << i << " name: " << input_name_str << std::endl;
        }

        // output names
        size_t num_output_nodes = onnx_session.GetOutputCount();
        std::vector<std::string> output_name_strings;
        std::vector<const char*> output_names_ptr;

        for (size_t i = 0; i < num_output_nodes; i++) {
            Ort::AllocatedStringPtr output_name_ptr = onnx_session.GetOutputNameAllocated(i, allocator);
            if (const char* output_name = output_name_ptr.get(); output_name && *output_name) {
                std::string output_name_str(output_name);
                output_name_strings.push_back(output_name_str);
                std::cout << "Output " << i << " name: " << output_name_str << std::endl;
            } else {
                std::cerr << "Output name cannot be empty" << std::endl;
                return 1;
            }
        }

        // string vector to char* vector for inference
        input_names_ptr.reserve(input_name_strings.size());
        for (const auto& name : input_name_strings) {
            input_names_ptr.push_back(name.c_str());
        }
        output_names_ptr.reserve(output_name_strings.size());
        for (const auto& name : output_name_strings) {
            output_names_ptr.push_back(name.c_str());
        }

        // run inference using the names from above
        std::vector<Ort::Value> output_tensors = onnx_session.Run(
            Ort::RunOptions{}, input_names_ptr.data(), &input_tensor, 1, output_names_ptr.data(), output_names_ptr.size());

        // just printing it for simplicity
        const auto* output_data = output_tensors[0].GetTensorMutableData<float>();
        for (int i = 0; i < 10; ++i) {
            std::cout << "Output " << i << ": " << output_data[i] << std::endl;
        }
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime exception: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception occurred" << std::endl;
        return 1;
    }

    return 0;
}