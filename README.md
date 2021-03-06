# Dependencies

Note that the version requirements are what I have tested with, lower versions may (or may not) work.

* CUDA 9.2+ (cuRAND)
* Boost 1.68+
* Eigen 3.3+
* Libtiff
* Intel TBB 2019+
* GTest 1.8+ [Tests only]
* GMock 1.8+ [Tests only]
* JSON for Modern C++ (https://nlohmann.github.io/json) [Provided in external]
* TinyObjLoader (https://github.com/syoyo/tinyobjloader) [Provided in external]

# Compilation

## Requirements

* CMake 3.8+
* C++11 compatible compiler
* SM35 compatible GPU

## Instructions

```
mkdir build
cd build
cmake ../
make
```

## Options

* `WITH_PROGRAMS=ON/OFF`: Whether to compile the example programs in the `programs` folder.
* `WITH_TEST_ASSETS=ON/OFF`: Whether to download the example assets in the `assets` folder.
* `WITH_TESTS=ON/OFF`: Whether to compile the tests found in the `tests` folder.
