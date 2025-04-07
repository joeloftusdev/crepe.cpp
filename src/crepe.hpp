#ifndef CREPE_HPP
#define CREPE_HPP

#include <Eigen/Dense>
#include <onnxruntime_cxx_api.h>

namespace crepe
{
namespace constants
{
constexpr int SAMPLE_RATE = 16000;
constexpr int FFT_HOP = 160; // 10ms at 16kHz
constexpr int FRAME_LENGTH = 1024; // Window size for CREPE
constexpr float BASE_FREQUENCY = 10.0f;
constexpr float MODEL_BASE_CENTS = 1997.3794084376191f;
constexpr float MODEL_RANGE_CENTS = 7180.0f;
constexpr float MODEL_BINS = 360.0f;
constexpr float CENTS_CONVERSION = 1200.0f;
constexpr float FREQ_MIN = 10.0f; // approx min frequency (Hz)
constexpr float FREQ_MAX = 7900.0f; // approx max frequency (Hz)
constexpr float CENTER_OFFSET = 0.5f;
constexpr float OCTAVE_BASE = 2.0f;
constexpr float BINS_PER_OCTAVE = 120.0f;

constexpr int ONNX_THREADS = 1;
constexpr int ONNX_LOG_LEVEL = ORT_LOGGING_LEVEL_WARNING;

constexpr float CONFIDENCE_THRESHOLD = 0.5f;
constexpr int MIN_FRAMES = 5;
} // namespace constants

// Data structure for prediction results
struct PredictionResults
{
    Eigen::VectorXf pitches;
    Eigen::VectorXf confidences;
    Eigen::VectorXf times;
    int num_frames;
};

// Analysis results (derived statistics)
struct PredictionAnalytics
{
    float min_frequency = 0.0f;
    float max_frequency = 0.0f;
    float mean_confidence = 0.0f;
    float time_pitch_correlation = 0.0f;
    const PredictionResults *source_data = nullptr;
};

float calculate_correlation(const Eigen::Ref<const Eigen::VectorXf> &x,
                            const Eigen::Ref<const Eigen::VectorXf> &y);

float get_pitch_from_crepe(const float *output_data, size_t output_size);

void analyze_frequency_bins();

void normalize_audio(Eigen::Ref<Eigen::VectorXf> audio_vec);

PredictionResults run_inference(const std::vector<float> &audio_data, int sample_rate);

PredictionResults run_inference(const float *audio_data, int length, int sample_rate);

PredictionAnalytics calculate_analytics(const PredictionResults &results);

}

#endif //CREPE_HPP