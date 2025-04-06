#include "crepe.hpp"

#include <iostream>

extern const unsigned char model_ort_start[];
extern const size_t model_ort_size;

namespace crepe {
    float calculate_correlation(const Eigen::Ref<const Eigen::VectorXf> &x,
                                const Eigen::Ref<const Eigen::VectorXf> &y) {
        // Copies to modify
        Eigen::VectorXf x_tmp = x;
        Eigen::VectorXf y_tmp = y;

        // Normalize by removing mean
        x_tmp.array() -= x_tmp.mean();
        y_tmp.array() -= y_tmp.mean();

        // Return correlation coefficient
        return x_tmp.dot(y_tmp) / (x_tmp.norm() * y_tmp.norm());
    }

    float get_pitch_from_crepe(const float *output_data, const size_t output_size) {
        using namespace constants;

        const Eigen::Map<const Eigen::VectorXf> output(output_data, static_cast<Eigen::Index>(output_size));
        const std::ptrdiff_t max_index = std::distance(output.data(),
                                            std::max_element(output.data(),
                                                             output.data() + output_size));

        const float cents = MODEL_BASE_CENTS +
                            (static_cast<float>(max_index) * MODEL_RANGE_CENTS /
                             (MODEL_BINS - 1.0f));
        return BASE_FREQUENCY * std::pow(OCTAVE_BASE, cents / CENTS_CONVERSION);
    }

    void analyze_frequency_bins() {
        using namespace constants;

        std::cout << "CREPE Frequency Bin Analysis:" << std::endl;

        // Sample a few bin indices to check spacing
        const int bins[] = {0, 60, 120, 180, 240, 300, 359};

        std::cout << "Bin\tFrequency (Hz)" << std::endl;
        for (const int bin: bins) {
            const float freq = BASE_FREQUENCY *
                               std::pow(OCTAVE_BASE,
                                        (static_cast<float>(bin) - CENTER_OFFSET * MODEL_BINS) / BINS_PER_OCTAVE);
            std::cout << bin << "\t" << freq << std::endl;
        }

        std::cout << "Frequency = " << BASE_FREQUENCY << " * " << OCTAVE_BASE
                << "^((bin - " << CENTER_OFFSET << "*" << MODEL_BINS << ")/"
                << BINS_PER_OCTAVE << ")" << std::endl;
        std::cout << "This covers approximately " << FREQ_MIN << "Hz to "
                << FREQ_MAX << "Hz" << std::endl;
    }


    void normalize_audio(Eigen::Ref<Eigen::VectorXf> audio_vec) {
        // Remove dc offset
        const float mean = audio_vec.mean();
        audio_vec.array() -= mean;

        //normalize
        const float variance = audio_vec.squaredNorm() / static_cast<float>(audio_vec.size());

        if (const float std_dev = std::sqrt(variance); std_dev > 1e-10f) {
            // Avoid division by zero
            audio_vec /= std_dev;
        }
    }

    PredictionResults run_inference(const std::vector<float> &audio_data, const int sample_rate) {
        return run_inference(audio_data.data(), static_cast<int>(audio_data.size()), sample_rate);
    }

    PredictionResults run_inference(const float *audio_data, const int length, const int sample_rate) {
        using namespace constants;

        if (sample_rate != SAMPLE_RATE) {
            std::cout << "Warning: CREPE expects " << SAMPLE_RATE << "Hz audio, got "
                    << sample_rate << "Hz" << std::endl;
        }

        // initialize ONNX
        const Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "CREPE");

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(ONNX_THREADS);
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

        // Load the embedded model from model.ort.h
        Ort::Session onnx_session(env, model_ort_start, model_ort_size, session_options);

        // Map the audio data
        Eigen::Map<const Eigen::VectorXf> audio_eigen(audio_data, length);

        // Calculate number of frames
        const int num_frames = (length - FRAME_LENGTH) / FFT_HOP + 1;

        // Initialize results container
        PredictionResults results;
        results.pitches.resize(num_frames);
        results.confidences.resize(num_frames);
        results.times.resize(num_frames);
        results.num_frames = num_frames;

        // Get input/output names
        const Ort::AllocatorWithDefaultOptions allocator;
        const auto input_name_ptr = onnx_session.GetInputNameAllocated(0, allocator);
        const auto output_name_ptr = onnx_session.GetOutputNameAllocated(0, allocator);
        const char *input_name = input_name_ptr.get();
        const char *output_name = output_name_ptr.get();

        const Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const std::vector<int64_t> input_dims = {1, FRAME_LENGTH};

        // Pre-allocate frame buffer
        Eigen::VectorXf frame(FRAME_LENGTH);

        // Process each frame
        for (int i = 0; i < num_frames; i++) {
            const size_t start_idx = i * FFT_HOP;

            frame = audio_eigen.segment(static_cast<Eigen::Index>(start_idx), FRAME_LENGTH);

            normalize_audio(frame); // Normalize audio frame

            // Create input tensor
            const Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, frame.data(), FRAME_LENGTH,
                input_dims.data(), input_dims.size());

            // Run inference
            std::vector<Ort::Value> output_tensors = onnx_session.Run(
                Ort::RunOptions{}, &input_name, &input_tensor, 1, &output_name, 1);

            // Process results
            const auto *output_data = output_tensors[0].GetTensorMutableData<float>();
            const size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

            // Get pitch
            const float pitch = get_pitch_from_crepe(output_data, output_size);

            // Map the output to Eigen
            Eigen::Map<const Eigen::VectorXf> output_eigen(output_data, static_cast<Eigen::Index>(output_size));
            int max_index;
            const float confidence = output_eigen.maxCoeff(&max_index);

            // Store results
            results.pitches(i) = pitch;
            results.confidences(i) = confidence;
            results.times(i) = static_cast<float>(start_idx) / static_cast<float>(sample_rate);
        }

        return results;
    }

    PredictionAnalytics calculate_analytics(const PredictionResults &results) {
        PredictionAnalytics analytics;
        analytics.source_data = &results;

        //basic statistics
        analytics.mean_confidence = results.confidences.mean();
        analytics.min_frequency = results.pitches.minCoeff();
        analytics.max_frequency = results.pitches.maxCoeff();

        // Ccorrelation between time and pitch
        analytics.time_pitch_correlation = calculate_correlation(results.times, results.pitches);

        return analytics;
    }
} // namespace crepe
