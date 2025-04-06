#pragma once
#include <vector>
#include <string>
#include <Eigen/Dense>

std::vector<float> load_wav_file(const std::string& filename, int* out_sample_rate = nullptr, std::string* error_msg = nullptr);

void normalize_audio(Eigen::Ref<Eigen::VectorXf> audio_vec);