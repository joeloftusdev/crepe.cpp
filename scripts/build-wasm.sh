#!/bin/bash

# <https://github.com/olilarkin/ort-builder>

python onnxruntime/tools/ci_build/build.py \
--build_dir onnxruntime/build/wasm \
--config=MinSizeRel \
--build_wasm_static_lib \
--parallel \
--minimal_build \
--disable_ml_ops --disable_exceptions --disable_rtti \
--include_ops_by_config model.required_operators_and_types.config \
--enable_reduced_operator_type_support \
--cmake_extra_defines CMAKE_POLICY_VERSION_MINIMUM=3.5 \
--skip_tests
