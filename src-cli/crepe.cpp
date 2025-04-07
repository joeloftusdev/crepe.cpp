#include <iostream>
#include <vector>
#include <atomic>
#include <thread>
#include "../deps/queue/readerwriterqueue.h"

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
#include "crepe.hpp"


class AudioProcessor
{
private:
    moodycamel::ReaderWriterQueue<float> queue;
    std::atomic<bool> finished{false};
    size_t frame_size;

public:
    explicit AudioProcessor(const size_t size) : queue(size * 10), frame_size(size)
    {
    }

    void push(const float *data, const size_t count)
    {
        for (size_t i = 0; i < count; i++)
        {
            queue.enqueue(data[i]);
        }
    }

    bool get_frame(std::vector<float> &frame)
    {
        if (finished && queue.size_approx() < frame_size)
        {
            return false;
        }

        //until we have enough data
        while (queue.size_approx() < frame_size && !finished)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        if (queue.size_approx() < frame_size)
        {
            return false;
        }

        frame.resize(frame_size);
        for (size_t i = 0; i < frame_size; i++)
        {
            float value;
            if (bool success = queue.try_dequeue(value); !success)
                return false;
            frame[i] = value;
        }
        return true;
    }

    void set_finished()
    {
        finished = true;
    }
};

//miniaudio callbacl
void data_callback(ma_device *device, void *output, const void *input, ma_uint32 frame_count)
{
    auto *processor = static_cast<AudioProcessor *>(device->pUserData);
    const auto *in_samples = static_cast<const float *>(input);

    //push
    processor->push(in_samples, frame_count * device->capture.channels);

    // clear if it's required
    if (output != nullptr)
    {
        memset(output, 0, frame_count * device->playback.channels * sizeof(float));
    }
}

int main()
{
    try
    {
        //init miniaudio
        ma_device_config config = ma_device_config_init(ma_device_type_capture);
        config.capture.format = ma_format_f32;
        config.capture.channels = 1; // Mono for simplicity
        config.sampleRate = crepe::constants::SAMPLE_RATE;
        config.dataCallback = data_callback;

        AudioProcessor processor(crepe::constants::FRAME_LENGTH);
        config.pUserData = &processor;

        ma_device device;
        if (ma_device_init(nullptr, &config, &device) != MA_SUCCESS)
        {
            std::cerr << "Error: Failed to initialize audio device" << std::endl;
        }

        if (ma_device_start(&device) != MA_SUCCESS)
        {
            ma_device_uninit(&device);
            std::cerr << "Error: Failed to start audio device" << std::endl;
        }

        std::cout << "Recording. Press Enter to stop." << std::endl;

        //analysis thread
        std::atomic<bool> running{true};
        std::thread analysis_thread([&]() {
            std::vector<float> frame;
            while (running)
            {
                if (processor.get_frame(frame))
                {
                    // Run inference on the frame
                    // Display the detected pitch
                    if (crepe::PredictionResults results = crepe::run_inference(
                        frame, crepe::constants::SAMPLE_RATE); results.num_frames > 0)
                    {
                        std::cout << "Pitch: " << results.pitches(0) << " Hz, Confidence: "
                            << results.confidences(0) << "   \r" << std::flush;
                    }
                }
            }
        });

        // Wait for user to press Enter
        std::cin.get();

        running = false;
        processor.set_finished();

        if (analysis_thread.joinable())
        {
            analysis_thread.join();
        }

        ma_device_uninit(&device);
        std::cout << "\nRecording stopped." << std::endl;

    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}