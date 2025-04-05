#include "wavloader.h"
#include <fstream>
#include <cstring>


// simple function to load a wav file. I should really replace this with a library like libsndfile/miniaudio
std::vector<float> load_wav_file(const std::string& filename, std::string* error_msg) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        if (error_msg) *error_msg = "Could not open WAV file: " + filename;
        return {};
    }

    struct WavHeader {
        char riff_id[4];
        int riff_size;
        char wave_id[4];
        char fmt_id[4];
        int fmt_size;
        short audio_format;
        short num_channels;
        int sample_rate;
        int byte_rate;
        short block_align;
        short bits_per_sample;
        char data_id[4];
        int data_size;
    };

    WavHeader header{};
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (std::strncmp(header.riff_id, "RIFF", 4) != 0 ||
        std::strncmp(header.wave_id, "WAVE", 4) != 0 ||
        std::strncmp(header.fmt_id, "fmt ", 4) != 0 ||
        std::strncmp(header.data_id, "data", 4) != 0) {
        if (error_msg) *error_msg = "Invalid WAV file format";
        return {};
        }

    if (header.audio_format != 1) {
        if (error_msg) *error_msg = "Only PCM format is supported";
        return {};
    }

    if (header.bits_per_sample != 16) {
        if (error_msg) *error_msg = "Only 16-bit audio is supported";
        return {};
    }

    std::vector<short> audio_data(header.data_size / sizeof(short));
    file.read(reinterpret_cast<char*>(audio_data.data()), header.data_size);

    std::vector<float> float_audio_data(audio_data.size());
    for (size_t i = 0; i < audio_data.size(); ++i) {
        float_audio_data[i] = static_cast<float>(audio_data[i]) / 32768.0f; // mormalize
    }

    return float_audio_data;
}