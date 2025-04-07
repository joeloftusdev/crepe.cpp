#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "crepe.hpp"
#define MINIAUDIO_IMPLEMENTATION
#include "../deps/miniaudio/miniaudio.h"
#include <iostream>

std::vector<float> load_wav_file(const std::string &filename, int *out_sample_rate, std::string *error_msg) {
    ma_decoder decoder;

    if (const ma_result result = ma_decoder_init_file(filename.c_str(), nullptr, &decoder); result != MA_SUCCESS) {
        if (error_msg) *error_msg = "Failed to initialize decoder for file: " + filename;
        return {};
    }

    if (out_sample_rate) {
        *out_sample_rate = static_cast<int>(decoder.outputSampleRate);
        std::cout << "Debug: Detected sample rate: " << *out_sample_rate << "Hz" << std::endl;
    }

    ma_uint64 frame_count;
    ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count);


    if (out_sample_rate) {
        *out_sample_rate = decoder.outputSampleRate;
    }

    const bool needs_conversion = decoder.outputChannels > 1;

    std::vector<float> audio_data(frame_count);

    if (needs_conversion) {
        std::vector<float> multi_channel_data(frame_count * decoder.outputChannels);
        const ma_uint64 frames_read =
                ma_decoder_read_pcm_frames(&decoder, multi_channel_data.data(), frame_count, nullptr);

        for (ma_uint64 i = 0; i < frames_read; i++) {
            float sum = 0.0f;
            for (ma_uint32 c = 0; c < decoder.outputChannels; c++) {
                sum += multi_channel_data[i * decoder.outputChannels + c];
            }
            audio_data[i] = sum / decoder.outputChannels;
        }
    } else {
        ma_decoder_read_pcm_frames(&decoder, audio_data.data(), frame_count, nullptr);
    }

    ma_decoder_uninit(&decoder);
    return audio_data;
}

int main() {
    try {
        // enable this to debug frequency mapping
        //crepe::analyze_frequency_bins();

        const std::string wav_file_path = "sweep.wav";
        std::ifstream file_check(wav_file_path);
        if (!file_check) {
            std::cerr << "Error: Cannot open file at path: " << wav_file_path << std::endl;
            return 1;
        }
        file_check.close();

        int sample_rate = 0;
        const std::vector<float> audio_data = load_wav_file(wav_file_path, &sample_rate, nullptr);

        // Run inference using the function from inference.cpp
        crepe::PredictionResults results = crepe::run_inference(audio_data, sample_rate);

        // Calculate analytics
        const crepe::PredictionAnalytics analytics = crepe::calculate_analytics(results);

        std::cout << "\nResults Summary:" << std::endl;
        std::cout << "Processed " << results.num_frames << " frames" << std::endl;
        std::cout << "Mean confidence: " << analytics.mean_confidence << std::endl;

        std::cout << "Sample frequencies (Hz): [";
        for (int i = 0; i < std::min(5, results.num_frames); i++) {
            std::cout << results.pitches(i);
            if (i < std::min(4, results.num_frames - 1)) std::cout << " ";
        }
        std::cout << "]" << std::endl;

        std::cout << "Min frequency: " << analytics.min_frequency << std::endl;
        std::cout << "Max frequency: " << analytics.max_frequency << std::endl;
        std::cout << "Correlation between time and frequency: " << analytics.time_pitch_correlation << std::endl;
        std::cout << "Should be close to 1.0 for frequency sweep" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}