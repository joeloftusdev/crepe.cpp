#include "crepe.hpp"

#include <iostream>

extern const unsigned char model_ort_start[];
extern const size_t model_ort_size;

namespace crepe
{
// precomputed constants
namespace constants_precomputed
{
    constexpr float PITCH_CONVERSION_FACTOR = constants::MODEL_RANGE_CENTS / (constants::MODEL_BINS - 1.0f);
    constexpr float OCTAVE_FACTOR = 1.0f / constants::CENTS_CONVERSION;
}

float calculate_correlation(const Eigen::Ref<const Eigen::VectorXf> &x,
                          const Eigen::Ref<const Eigen::VectorXf> &y)
{
    const float x_mean = x.mean();
    const float y_mean = y.mean();

    float numerator = 0.0f;
    float x_norm_sq = 0.0f;
    float y_norm_sq = 0.0f;

    for (Eigen::Index i = 0; i < x.size(); ++i) {
        const float x_diff = x(i) - x_mean;
        const float y_diff = y(i) - y_mean;
        numerator += x_diff * y_diff;
        x_norm_sq += x_diff * x_diff;
        y_norm_sq += y_diff * y_diff;
    }

    return numerator / (std::sqrt(x_norm_sq) * std::sqrt(y_norm_sq));
}

float get_pitch_from_crepe(const float *output_data, const size_t output_size)
{
    using namespace constants;
    using namespace constants_precomputed;

    const Eigen::Map<const Eigen::VectorXf> output(output_data,
                                                 static_cast<Eigen::Index>(output_size));
    Eigen::Index max_index;
    output.maxCoeff(&max_index);

    const float cents = MODEL_BASE_CENTS + (static_cast<float>(max_index) * PITCH_CONVERSION_FACTOR);
    return BASE_FREQUENCY * std::pow(OCTAVE_BASE, cents * OCTAVE_FACTOR);
}

void analyze_frequency_bins()
{
    using namespace constants;

    std::cout << "CREPE Frequency Bin Analysis:" << std::endl;

    // Sample a few bin indices to check spacing
    const int bins[] = {0, 60, 120, 180, 240, 300, 359};

    std::cout << "Bin\tFrequency (Hz)" << std::endl;
    for (const int bin : bins)
    {
        const float freq = BASE_FREQUENCY *
                         std::pow(OCTAVE_BASE,
                                (static_cast<float>(bin) - CENTER_OFFSET * MODEL_BINS) /
                                BINS_PER_OCTAVE);
        std::cout << bin << "\t" << freq << std::endl;
    }

    std::cout << "Frequency = " << BASE_FREQUENCY << " * " << OCTAVE_BASE
        << "^((bin - " << CENTER_OFFSET << "*" << MODEL_BINS << ")/"
        << BINS_PER_OCTAVE << ")" << std::endl;
    std::cout << "This covers approximately " << FREQ_MIN << "Hz to "
        << FREQ_MAX << "Hz" << std::endl;
}


void normalize_audio(Eigen::Ref<Eigen::VectorXf> audio_vec)
{
    // Remove dc offset
    const float mean = audio_vec.mean();
    audio_vec.array() -= mean;

    //normalize
    const float variance = audio_vec.squaredNorm() / static_cast<float>(audio_vec.size());

    if (const float std_dev = std::sqrt(variance); std_dev > 1e-10f)
    {
        const float inv_std_dev = 1.0f / std_dev;
        audio_vec *= inv_std_dev;
    }
}

class CrepeModel {
private:
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::string input_name;
    std::string output_name;
    Ort::MemoryInfo memory_info;
    std::vector<int64_t> input_dims;

    CrepeModel() :
        env(ORT_LOGGING_LEVEL_WARNING, "CREPE"),
        session(nullptr),
        memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
        input_dims({1, constants::FRAME_LENGTH})
    {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(constants::ONNX_THREADS);
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

        session = Ort::Session(env, model_ort_start, model_ort_size, session_options);

        const auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
        const auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);

        input_name = input_name_ptr.get();
        output_name = output_name_ptr.get();
    }

public:
    CrepeModel(const CrepeModel&) = delete;
    CrepeModel& operator=(const CrepeModel&) = delete;
    CrepeModel(CrepeModel&&) = delete;
    CrepeModel& operator=(CrepeModel&&) = delete;

    static CrepeModel& getInstance() {
        static CrepeModel instance; // automatically destroyed
        return instance;
    }

    PredictionResults runInference(const float* audio_data, int length, int sample_rate);
};

//CrepeModel* CrepeModel::instance = nullptr;

PredictionResults CrepeModel::runInference(const float* audio_data, int length, int sample_rate)
{
    using namespace constants;

    if (sample_rate != SAMPLE_RATE)
    {
        std::cout << "Warning: CREPE expects " << SAMPLE_RATE << "Hz audio, got "
            << sample_rate << "Hz" << std::endl;
    }

    Eigen::Map<const Eigen::VectorXf> audio_eigen(audio_data, length);

    const int num_frames = (length - FRAME_LENGTH) / FFT_HOP + 1;

    PredictionResults results;
    results.pitches.resize(num_frames);
    results.confidences.resize(num_frames);
    results.times.resize(num_frames);
    results.num_frames = num_frames;

    Eigen::VectorXf frame(FRAME_LENGTH);

    // process each frame in parallel
    #pragma omp parallel for if(num_frames > 16) private(frame)
    for (int i = 0; i < num_frames; i++)
    {
        const size_t start_idx = i * FFT_HOP;

        frame = audio_eigen.segment(static_cast<Eigen::Index>(start_idx), FRAME_LENGTH);
        normalize_audio(frame); // Normalize

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, frame.data(), FRAME_LENGTH,
            input_dims.data(), input_dims.size());

        //  inference -  use array of pointers for input/output names
        const char* input_names[] = {input_name.c_str()};
        const char* output_names[] = {output_name.c_str()};
        std::vector<Ort::Value> output_tensors = session.Run(
            Ort::RunOptions{}, input_names, &input_tensor, 1, output_names, 1);

        // results
        const auto *output_data = output_tensors[0].GetTensorMutableData<float>();
        const size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

        // pitch
        const float pitch = get_pitch_from_crepe(output_data, output_size);

        // Map the output to Eigen
        Eigen::Map<const Eigen::VectorXf> output_eigen(output_data,
                                                   static_cast<Eigen::Index>(output_size));
        int max_index;
        const float confidence = output_eigen.maxCoeff(&max_index);

        // Store results
        #pragma omp critical // prevent race
        {
            results.pitches(i) = pitch;
            results.confidences(i) = confidence;
            results.times(i) = static_cast<float>(start_idx) / static_cast<float>(sample_rate);
        }
    }

    return results;
}

PredictionResults run_inference(const std::vector<float> &audio_data, const int sample_rate)
{
    return run_inference(audio_data.data(), static_cast<int>(audio_data.size()), sample_rate);
}

PredictionResults run_inference(const float *audio_data, const int length, const int sample_rate)
{
    return CrepeModel::getInstance().runInference(audio_data, length, sample_rate);
}

PredictionAnalytics calculate_analytics(const PredictionResults &results)
{
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