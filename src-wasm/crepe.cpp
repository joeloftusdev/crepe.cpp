#include <emscripten.h>
#include <emscripten/bind.h>

#include "crepe.hpp"
#include <vector>

// c interface
extern "C"
{
EMSCRIPTEN_KEEPALIVE
float *analyse_audio(const float *audio_data, const int length)
{
    // results
    auto [pitches, confidences, times, num_frames] = crepe::run_inference(
        audio_data, length, crepe::constants::SAMPLE_RATE);

    // allocate
    auto *result_array = static_cast<float *>(malloc((1 + num_frames * 3) * sizeof(float)));

    // number of frames
    result_array[0] = static_cast<float>(num_frames);

    // pitch, confidence, time for each frame
    for (int i = 0; i < num_frames; i++)
    {
        result_array[1 + i * 3] = pitches(i);
        result_array[1 + i * 3 + 1] = confidences(i);
        result_array[1 + i * 3 + 2] = times(i);
    }

    return result_array;
}
}

// or js binding
emscripten::val analysePitch(const emscripten::val &audio_array)
{
    // convert js array to std::vector<float>
    std::vector<float> audio_data;
    const auto length = audio_array["length"].as<unsigned>();
    audio_data.reserve(length);

    for (unsigned i = 0; i < length; i++)
    {
        audio_data.push_back(audio_array[i].as<float>());
    }

    // inference
    auto [pitches, confidences, times, num_frames] = crepe::run_inference(
        audio_data.data(), audio_data.size(), crepe::constants::SAMPLE_RATE);

    // js array of objects
    emscripten::val result = emscripten::val::array();

    for (int i = 0; i < num_frames; i++)
    {
        emscripten::val frame = emscripten::val::object();
        frame.set("pitch", pitches(i));
        frame.set("confidence", confidences(i));
        frame.set("time", times(i));
        result.call<void>("push", frame);
    }

    return result;
}

EMSCRIPTEN_BINDINGS(crepe_module)
{
    emscripten::function("analysePitch", &analysePitch);
}