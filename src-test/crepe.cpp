#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <vector>
#include "crepe.hpp"
#include "../deps/miniaudio/miniaudio.h"

std::vector<float> load_wav_file(const std::string &filename, int *out_sample_rate,
                                 std::string *error_msg)
{
    ma_decoder decoder;

    if (const ma_result result = ma_decoder_init_file(filename.c_str(), nullptr, &decoder);
        result != MA_SUCCESS)
    {
        if (error_msg)
            *error_msg = "Failed to initialize decoder for file: " + filename;
        return {};
    }

    if (out_sample_rate)
    {
        *out_sample_rate = static_cast<int>(decoder.outputSampleRate);
    }

    ma_uint64 frame_count;
    ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count);

    const bool needs_conversion = decoder.outputChannels > 1;

    std::vector<float> audio_data(frame_count);

    if (needs_conversion)
    {
        std::vector<float> multi_channel_data(frame_count * decoder.outputChannels);
        const ma_uint64 frames_read =
            ma_decoder_read_pcm_frames(&decoder, multi_channel_data.data(), frame_count, nullptr);

        for (ma_uint64 i = 0; i < frames_read; i++)
        {
            float sum = 0.0f;
            for (ma_uint32 c = 0; c < decoder.outputChannels; c++)
            {
                sum += multi_channel_data[i * decoder.outputChannels + c];
            }
            audio_data[i] = sum / decoder.outputChannels;
        }
    }
    else
    {
        ma_decoder_read_pcm_frames(&decoder, audio_data.data(), frame_count, nullptr);
    }

    ma_decoder_uninit(&decoder);
    return audio_data;
}

TEST_CASE("Frequency Analysis with CREPE - Detailed Output", "[crepe]") {
    const std::string wav_file_path = "sweep.wav";

    std::ifstream file_check(wav_file_path);
    REQUIRE(file_check.good());
    file_check.close();

    int sample_rate = 0;
    std::string error_msg;
    const std::vector<float> audio_data = load_wav_file(wav_file_path, &sample_rate, &error_msg);
    REQUIRE(!audio_data.empty());
    INFO("Sample rate: " << sample_rate << "Hz");

    SECTION("Run inference and validate results with detailed output") {
        crepe::PredictionResults results = crepe::run_inference(audio_data, sample_rate);
        REQUIRE(results.num_frames > 0);

        const crepe::PredictionAnalytics analytics = crepe::calculate_analytics(results);

        INFO("Results Summary:");
        INFO("Processed " << results.num_frames << " frames");
        INFO("Mean confidence: " << analytics.mean_confidence);

        std::stringstream freq_stream;
        freq_stream << "Sample frequencies (Hz): [";
        for (int i = 0; i < std::min(5, results.num_frames); i++) {
            freq_stream << results.pitches(i);
            if (i < std::min(4, results.num_frames - 1))
                freq_stream << " ";
        }
        freq_stream << "]";
        INFO(freq_stream.str());

        INFO("Min frequency: " << analytics.min_frequency);
        INFO("Max frequency: " << analytics.max_frequency);
        INFO("Correlation between time and frequency: " << analytics.time_pitch_correlation);
        INFO("Should be close to 1.0 for frequency sweep");

        CHECK(analytics.mean_confidence > 0.0f);
        CHECK(analytics.min_frequency > 0.0f);
        CHECK(analytics.max_frequency > analytics.min_frequency);
        CHECK(analytics.time_pitch_correlation > 0.9f);
    }
}