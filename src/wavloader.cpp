#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
#include "wavloader.h"
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
                ma_decoder_read_pcm_frames(&decoder, multi_channel_data.data(), frame_count, NULL);

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




void normalize_audio(Eigen::Ref<Eigen::VectorXf> audio_vec) {
    // Remove dc offset
    const float mean = audio_vec.mean();
    audio_vec.array() -= mean;

    //normalize
    const float variance = audio_vec.squaredNorm() / static_cast<float>(audio_vec.size());

    if (const float std_dev = std::sqrt(variance); std_dev > 1e-10f) {  // Avoid division by zero
        audio_vec /= std_dev;
    }
}