#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>


#include "../external/model.ort.h"
#include "wavloader.h"

float calculate_correlation(const Eigen::Ref<const Eigen::VectorXf>& x,
                           const Eigen::Ref<const Eigen::VectorXf>& y) {
    //copies to modify
    Eigen::VectorXf x_tmp = x;
    Eigen::VectorXf y_tmp = y;

    x_tmp.array() -= x_tmp.mean();
    y_tmp.array() -= y_tmp.mean();

    return x_tmp.dot(y_tmp) / (x_tmp.norm() * y_tmp.norm());
}

float get_pitch_from_crepe(const float* output_data, const size_t output_size) {
    const Eigen::Map<const Eigen::VectorXf> output(output_data, output_size);
    const int max_index = std::distance(output.data(),
                                      std::max_element(output.data(),
                                      output.data() + output_size));

    const float cents = 1997.3794084376191f + (static_cast<float>(max_index) * 7180.0f / 359.0f);
    return 10.0f * std::pow(2.0f, cents / 1200.0f);
}

void analyze_frequency_bins() {
    std::cout << "CREPE Frequency Bin Analysis:" << std::endl;

    // Sample a few bin indices to check spacing
    const int bins[] = {0, 60, 120, 180, 240, 300, 359};

    std::cout << "Bin\tFrequency (Hz)" << std::endl;
    for (const int bin : bins) {
        const float freq = 10.0f * std::pow(2.0f, (static_cast<float>(bin) - 0.5f * 360.0f) / 120.0f);
        std::cout << bin << "\t" << freq << std::endl;
    }

    std::cout << "Frequency = 10.0 * 2^((bin - 0.5*360)/120)" << std::endl;
    std::cout << "This covers approximately 10Hz to 7.9kHz" << std::endl;
}

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

        // Map the audio data
        Eigen::Map<Eigen::VectorXf> audio_eigen(audio_data.data(), audio_data.size());

        constexpr int hop_length = 160; // Match python impl default (10ms at 16kHz)
        constexpr int frame_length = 1024;

        // result containers
        const int num_frames = (audio_data.size() - frame_length) / hop_length + 1;
        Eigen::VectorXf pitches(num_frames);
        Eigen::VectorXf confidences(num_frames);
        Eigen::VectorXf times(num_frames);

        // get input/output names
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name_ptr = onnx_session.GetInputNameAllocated(0, allocator);
        auto output_name_ptr = onnx_session.GetOutputNameAllocated(0, allocator);
        const char* input_name = input_name_ptr.get();
        const char* output_name = output_name_ptr.get();

        std::cout << "Using input name: " << input_name << std::endl;
        std::cout << "Using output name: " << output_name << std::endl;

        const Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const std::vector<int64_t> input_dims = {1, frame_length};

        // pre-allocate frame buffer
        Eigen::VectorXf frame(frame_length);

        for (int i = 0; i < num_frames; i++) {
            const size_t start_idx = i * hop_length;

            frame = audio_eigen.segment(start_idx, frame_length);

            normalize_audio(frame); //normalize

            // create input tensor
            const Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, frame.data(), frame_length,
                input_dims.data(), input_dims.size());

            // running inference
            std::vector<Ort::Value> output_tensors = onnx_session.Run(
                Ort::RunOptions{}, &input_name, &input_tensor, 1, &output_name, 1);

            // Process results
            const auto* output_data = output_tensors[0].GetTensorMutableData<float>();
            const size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

            // Get pitch and confidence
            float pitch = get_pitch_from_crepe(output_data, output_size);

            // Map the output toEigen
            Eigen::Map<const Eigen::VectorXf> output_eigen(output_data, output_size);
            int max_index;
            float confidence = output_eigen.maxCoeff(&max_index);

            // Store results
            pitches(i) = pitch;
            confidences(i) = confidence;
            times(i) = static_cast<float>(start_idx) / static_cast<float>(sample_rate);

            // Print just the first few frames
            if (i < 5) {
                std::cout << "Frame " << (i + 1) << ": "
                          << pitch << " Hz (confidence: " << confidence << ")" << std::endl;
            }
        }

        // Calculate statistics similar to the Python test
        std::cout << "\nResults Summary:" << std::endl;
        std::cout << "Processed " << pitches.size() << " frames" << std::endl;

        float mean_confidence = confidences.mean();
        std::cout << "Mean confidence: " << mean_confidence << std::endl;

        std::cout << "Sample frequencies (Hz): [";
        for (int i = 0; i < std::min(5, num_frames); i++) {
            std::cout << pitches(i);
            if (i < std::min(4, num_frames - 1)) std::cout << " ";
        }
        std::cout << "]" << std::endl;

        float min_freq = pitches.minCoeff();
        float max_freq = pitches.maxCoeff();
        std::cout << "Min frequency: " << min_freq << std::endl;
        std::cout << "Max frequency: " << max_freq << std::endl;

        float times_mean = times.mean();
        float pitches_mean = pitches.mean();

        Eigen::VectorXf times_centered = times;
        times_centered.array() -= times_mean;

        Eigen::VectorXf pitches_centered = pitches;
        pitches_centered.array() -= pitches_mean;

        float correlation = times_centered.dot(pitches_centered) /
                            (times_centered.norm() * pitches_centered.norm());

        std::cout << "Correlation between time and frequency: " << correlation << std::endl;
        std::cout << "Should be close to 1.0 for frequency sweep" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}


