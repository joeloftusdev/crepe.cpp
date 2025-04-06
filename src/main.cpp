#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "wavloader.h"
#include "crepe.hpp"

int main() {
    try {
        // enable this to debug frequency mapping
        //crepe::analyze_frequency_bins();

        const std::string wav_file_path = "../external/sweep.wav";
        std::ifstream file_check(wav_file_path);
        if (!file_check) {
            std::cerr << "Error: Cannot open file at path: " << wav_file_path << std::endl;
            return 1;
        }
        file_check.close();

        int sample_rate = 0;
        const std::vector<float> audio_data = load_wav_file(wav_file_path, &sample_rate);

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