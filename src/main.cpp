#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "../external/model.ort.h"
#include "wavloader.h"

float get_pitch_from_crepe(const float* output_data, size_t output_size) {
    int max_index = 0;
    float max_value = output_data[0];

    for (size_t i = 1; i < output_size; ++i) {
        if (output_data[i] > max_value) {
            max_value = output_data[i];
            max_index = i;
        }
    }

    // Convert bin index to frequency (Hz)
    // CREPE maps indices to frequencies using this formula:
    // freq = 10.0 * 2^((indices - 0.5 * 360) / 120)
    float frequency = 10.0f * std::pow(2.0f, (max_index - 0.5f * 360.0f) / 120.0f);

    return frequency;
}

int main() {
    try {
        const Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "CREPE");

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

        // Load the model from the model.ort.h
        Ort::Session onnx_session(env, model_ort_start, model_ort_size, session_options);

        // Load audio data from WAV file
        std::string wav_file_path = "external/sweep.wav";
        std::vector<float> audio_data = load_wav_file(wav_file_path);

        // Check if the loaded audio is at least 1024 samples long.  If not, pad with zeros.
        if (audio_data.size() < 1024) {
            audio_data.resize(1024, 0.0f); // Pad with zeros
        }

        // Take the first 1024 samples
        std::vector<float> input_data(audio_data.begin(), audio_data.begin() + 1024);

        // expected dimensions for the input tensor
        const std::vector<int64_t> input_dims = {1, 1024};

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
        float pitch = get_pitch_from_crepe(output_data, output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount());
        std::cout << "Estimated pitch: " << pitch << " Hz" << std::endl;
        std::cout << "Confidence: " << output_data[std::distance(output_data, std::max_element(output_data, output_data + output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount()))] << std::endl;

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