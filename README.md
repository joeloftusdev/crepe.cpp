# crepe.cpp

C++20 inference for [CREPE](https://github.com/marl/crepe) a monophonic pitch tracker based on a deep convolutional neural network operating directly on the time-domain waveform input.

This uses [ONNXRuntime](https://github.com/microsoft/onnxruntime) & scripts from [ort-builder](https://github.com/olilarkin/ort-builder) to:
- Converts an ONNX model to ORT format and serializes it to C++ source code, generate custom slimmed ONNX Runtime static libs.
- Create a flexible but tiny inference engine for a specific model.
- A minimal ORT binary using only the CPU provider

# Design

* [crepe-model](./crepe-model) contains the ONNX & ORT model along with the generated h and c file
* [scripts](./scripts) contain the ORT model build scripts
* [src](./src) shared inference
* [src_wasm](./src-wasm) main WASM, for the web app
* [src_cli](./src-cli) is a very simple cli app that uses [miniaudio](https://github.com/mackron/miniaudio) for audio processing
* [src_test](./src-test) a simple test that replicates the orignal repo python test for debugging using [miniaudio](https://github.com/mackron/miniaudio)
* [deps](./deps) project dependencies
* [web](./web) Javascript/HTML code for the WASM app.


# Python

You can use my fork of [crepe](https://github.com/joeloftusdev/crepe/tree/master), this adds ONNX export capability for CREPE models

- export_model_to_onnx() function to core.py that converts Keras models to ONNX
- A script (onnx_export.py) to easily export models with different capacities

Create your venv and get your requirements.

Usage examples:
```
$ pip install -e .
$ python -m crepe.onnx_export tiny  # Creates model-tiny.onnx
$ python -m crepe.onnx_export full -o custom_name.onnx
```

Then we will use [ort-builder](https://github.com/olilarkin/ort-builder)  : 
```
$ git submodule update --init
```
```
$ python3 -m venv venv
$ source ./venv/bin/activate`

$ pip install -r requirements.txt
$ ./convert-model-to-ort.sh model.onnx
```

Now we build our customized onnx runtime static libraries

```mac
$ ./build-mac.sh
```



# Build 

On macOS. Assuming that you have a typical C++ toolchain, CMake and AppleClang/clang etc.
You're also going to need to set up the [Emscripten SDK](https://github.com/emscripten-core/emsdk) for compiling to WebAssembly.

```
$ git clone --recurse-submodules https://github.com/joeloftusdev/crepe.cpp
```

cli/test
```
$ mkdir build
$ cd build

$ cmake -DCMAKE_BUILD_TYPE=Release ..

$ cmake --build .
```

Example output running the catch2 test: 

```
# Run the test with verbose output:
$ ./src-test/crepe_test -s
```

```
PASSED:
  CHECK( analytics.mean_confidence > 0.0f )
with expansion:
  0.84595f > 0.0f
with messages:
  Sample rate: 16000Hz
  Results Summary:
  Processed 270 frames
  Mean confidence: 0.845953
  Sample frequencies (Hz): [187.803 187.803 189.985 192.193 192.193]
  Min frequency: 187.803
  Max frequency: 1766.17
  Correlation between time and frequency: 0.961408
  Should be close to 1.0 for frequency sweep
```

wasm:
```
$ source "/path/to/emsdk/emsdk_env.sh"
$ emcmake cmake -S . -B build-wasm-release -DCMAKE_BUILD_TYPE=Release
$ emmake cmake --build build-wasm-release   
```

Open the index.html with live server or whatever you preference and view the wasm app:

<div align="center">
  <img src="https://github.com/user-attachments/assets/cdf9317f-770b-4376-8f3e-da5a8afa613e" alt="Screenshot" width="500">
</div>


# Credit

- The orignal [CREPE](https://github.com/marl/crepe) repo
- [ort-builder](https://github.com/olilarkin/ort-builder)
- [basicpitch.cpp](https://github.com/sevagh/basicpitch.cpp) was a great reference for the structure of this repo.


