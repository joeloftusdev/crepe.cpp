#pragma once
#include <vector>
#include <string>

std::vector<float> load_wav_file(const std::string& filename, std::string* error_msg = nullptr);