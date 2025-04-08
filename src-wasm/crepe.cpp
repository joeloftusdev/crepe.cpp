#include <emscripten.h>
#include <emscripten/bind.h>

#include "crepe.hpp"
#include <vector>
#include <cstring> 

namespace {
    constexpr size_t MAX_AUDIO_SAMPLES = 16000; 
    constexpr size_t MAX_FRAMES = 100;

    float g_audio_buffer[MAX_AUDIO_SAMPLES];
    float g_result_buffer[1 + MAX_FRAMES * 3];
    std::vector<float> g_audio_vector;
}


extern "C" {
EMSCRIPTEN_KEEPALIVE
float* analyse_audio(const float* audio_data, const int length) {
    const float* inference_data;
    const int valid_length = std::min(length, static_cast<int>(MAX_AUDIO_SAMPLES));

    if (length <= MAX_AUDIO_SAMPLES) {
        inference_data = audio_data;
    } else {
        memcpy(g_audio_buffer, audio_data, valid_length * sizeof(float));
        inference_data = g_audio_buffer;
    }

    auto [pitches, confidences, times, num_frames] = crepe::run_inference(
        inference_data, valid_length, crepe::constants::SAMPLE_RATE);

    const int safe_frames = std::min(num_frames, static_cast<int>(MAX_FRAMES));
    g_result_buffer[0] = static_cast<float>(safe_frames);

    for (int i = 0; i < safe_frames; i++) {
        const int base_idx = 1 + i * 3;
        g_result_buffer[base_idx] = pitches(i);
        g_result_buffer[base_idx + 1] = confidences(i);
        g_result_buffer[base_idx + 2] = times(i);
    }

    return g_result_buffer;
}
} 

emscripten::val analysePitch(const emscripten::val& audio_array) {
    const auto length = audio_array["length"].as<unsigned>();
    float* result_ptr;

    if (audio_array.instanceof(emscripten::val::global("Float32Array"))) {
        const uintptr_t data_ptr = audio_array["byteOffset"].as<uintptr_t>() +
                                   emscripten::val::global("Module")["HEAPF32"]["byteOffset"].as<uintptr_t>();
        const float* audio_ptr = reinterpret_cast<const float*>(data_ptr);
        result_ptr = analyse_audio(audio_ptr, length);
    } else {
        g_audio_vector.resize(length);
        for (unsigned i = 0; i < length; i++) {
            g_audio_vector[i] = audio_array[i].as<float>();
        }
        result_ptr = analyse_audio(g_audio_vector.data(), length);
    }

    const int num_frames = static_cast<int>(result_ptr[0]);
    emscripten::val result = emscripten::val::array();

    for (int i = 0; i < num_frames; i++) {
        const int base_idx = 1 + i * 3;
        emscripten::val frame = emscripten::val::object();
        frame.set("pitch", result_ptr[base_idx]);
        frame.set("confidence", result_ptr[base_idx + 1]);
        frame.set("time", result_ptr[base_idx + 2]);
        result.call<void>("push", frame);
    }

    return result;
}

EMSCRIPTEN_BINDINGS(crepe_module) {
    emscripten::function("analysePitch", &analysePitch);
}