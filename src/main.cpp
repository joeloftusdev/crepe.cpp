#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "../external/model.ort.h"
#include "wavloader.h"

float calculate_correlation(const std::vector<float>& x, const std::vector<float>& y) {
    if (x.size() != y.size() || x.empty()) return 0.0f;

    float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f;
    float sum_x2 = 0.0f, sum_y2 = 0.0f;

    for (size_t i = 0; i < x.size(); i++) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
        sum_y2 += y[i] * y[i];
    }

    const auto n = static_cast<float>(x.size());
    return (n * sum_xy - sum_x * sum_y) /
           (std::sqrt(n * sum_x2 - sum_x * sum_x) * std::sqrt(n * sum_y2 - sum_y * sum_y));
}

float get_pitch_from_crepe(const float *output_data, const size_t output_size) {
    int max_index = 0;
    float max_value = output_data[0];

    for (size_t i = 1; i < output_size; ++i) {
        if (output_data[i] > max_value) {
            max_value = output_data[i];
            max_index = i;
        }
    }

    // Convert index to cents
    const float cents = 1997.3794084376191f + (static_cast<float>(max_index) * 7180.0f / 359.0f);
    // Convert cents to frequency
    const float frequency = 10.0f * std::pow(2.0f, cents / 1200.0f);
    return frequency;
}

// void analyze_frequency_bins() {
//     std::cout << "CREPE Frequency Bin Analysis:" << std::endl;
//
//     // Sample a few bin indices to check spacing
//     const int bins[] = {0, 60, 120, 180, 240, 300, 359};
//
//     std::cout << "Bin\tFrequency (Hz)" << std::endl;
//     for (const int bin : bins) {
//         const float freq = 10.0f * std::pow(2.0f, (bin - 0.5f * 360.0f) / 120.0f);
//         std::cout << bin << "\t" << freq << std::endl;
//     }
//
//     std::cout << "CREPE uses logarithmic frequency spacing (not linear)" << std::endl;
//     std::cout << "Frequency = 10.0 * 2^((bin - 0.5*360)/120)" << std::endl;
//     std::cout << "This covers approximately 10Hz to 7.9kHz" << std::endl;
// }

int main() {
    try {
        // Enable this to debug frequency mapping
        //analyze_frequency_bins();

        const Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "CREPE");

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

        // Load the model from the model.ort.h
        Ort::Session onnx_session(env, model_ort_start, model_ort_size, session_options);

        std::string wav_file_path = "../external/sweep.wav";
        std::ifstream file_check(wav_file_path);
        if (!file_check) {
            std::cerr << "Error: Cannot open file at path: " << wav_file_path << std::endl;
            return 1;
        }
        file_check.close();
        int sample_rate = 0;
        std::vector<float> audio_data = load_wav_file(wav_file_path, &sample_rate);

        if (sample_rate != 16000) {
            std::cout << "Warning: CREPE expects 16kHz audio, got " << sample_rate << "Hz" << std::endl;
        }

        // The original sweep is a linear frequency sweep from 20Hz to 2kHz
        // We need to process the audio in frames with overlap
        constexpr int hop_length = 160; // Match python impl default (10ms at 16kHz)
        constexpr int frame_length = 1024;

        std::vector<float> pitches;
        std::vector<float> confidences;
        std::vector<float> times;

        // Get input/output names once before the loop
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name_ptr = onnx_session.GetInputNameAllocated(0, allocator);
        auto output_name_ptr = onnx_session.GetOutputNameAllocated(0, allocator);
        const char* input_name = input_name_ptr.get();
        const char* output_name = output_name_ptr.get();

        std::cout << "Using input name: " << input_name << std::endl;
        std::cout << "Using output name: " << output_name << std::endl;

        for (size_t start_idx = 0; start_idx + frame_length <= audio_data.size(); start_idx += hop_length) {
            // Create a frame of audio
            std::vector frame(audio_data.begin() + start_idx,
                               audio_data.begin() + start_idx + frame_length);

            // Map to Eigen for normalization, if needed
            Eigen::Map<Eigen::VectorXf> frame_eigen(frame.data(), frame.size());
            normalize_audio(frame_eigen);

            // Set up input tensor
            const std::vector<int64_t> input_dims = {1, frame_length};
            const Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            const Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, frame.data(), frame.size(),
                input_dims.data(), input_dims.size());

            // Run inference with our previously obtained input/output names
            std::vector<Ort::Value> output_tensors = onnx_session.Run(
                Ort::RunOptions{}, &input_name, &input_tensor, 1, &output_name, 1);

            // Process results
            const auto* output_data = output_tensors[0].GetTensorMutableData<float>();
            const size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

            float pitch = get_pitch_from_crepe(output_data, output_size);

            // Find max confidence
            int max_index = std::distance(output_data,
                                        std::max_element(output_data, output_data + output_size));
            float confidence = output_data[max_index];

            // Store results
            pitches.push_back(pitch);
            confidences.push_back(confidence);
            times.push_back(static_cast<float>(start_idx) / sample_rate);

            //print just the first few frames
            if (pitches.size() <= 5) {
                std::cout << "Frame " << pitches.size() << ": "
                          << pitch << " Hz (confidence: " << confidence << ")" << std::endl;
            }
        }

        // Calculate statistics similar to the Python test
        std::cout << "\nResults Summary:" << std::endl;
        std::cout << "Processed " << pitches.size() << " frames" << std::endl;

        // Mean confidence
        float mean_confidence = 0.0f;
        for (float conf : confidences) mean_confidence += conf;
        mean_confidence /= confidences.size();
        std::cout << "Mean confidence: " << mean_confidence << std::endl;

        std::cout << "Sample frequencies (Hz): [";
        for (size_t i = 0; i < std::min(static_cast<size_t>(5), pitches.size()); i++) {
            std::cout << pitches[i];
            if (i < std::min(static_cast<size_t>(4), pitches.size() - 1)) std::cout << " ";
        }
        std::cout << "]" << std::endl;

        // Min and max
        float min_freq = *std::ranges::min_element(pitches);
        float max_freq = *std::ranges::max_element(pitches);
        std::cout << "Min frequency: " << min_freq << std::endl;
        std::cout << "Max frequency: " << max_freq << std::endl;

        float correlation = calculate_correlation(times, pitches);
        std::cout << "Correlation between time and frequency: " << correlation << std::endl;
        std::cout << "Should be close to 1.0 for frequency sweep" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}


