cmake_minimum_required(VERSION 3.14)

project(PyCuAmpcor LANGUAGES CXX) #CUDA)

include(GNUInstallDirs)

find_package(GDAL REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

#cython_add_module(PyCuAmpcor
    #src/PyCuAmpcor.pyx
add_library(cuampcor SHARED
    src/GDALImage.cpp
    src/cuAmpcorChunk.cpp
    src/cuAmpcorController.cpp
    src/cuAmpcorParameter.cpp
    src/cuArrays.cpp
    src/cuArraysCopy.cpp
    src/cuArraysPadding.cpp
    src/cuCorrFrequency.cpp
    src/cuCorrNormalization.cpp
    src/cuCorrTimeDomain.cpp
    src/cuDeramp.cpp
    src/cuEstimateStats.cpp
    src/cuOffset.cpp
    src/cuOverSampler.cpp
    src/cuSincOverSampler.cpp
    src/cudaError.cpp
    src/cudaUtil.cpp
    )
#[[
target_include_directories(PyCuAmpcor PRIVATE
    src
    )
target_link_libraries(PyCuAmpcor PRIVATE
    CUDA::cufft
    GDAL::GDAL
    Python::NumPy
    )

InstallSameDir(
    __init__.py
    PyCuAmpcor
    )
#]]

target_include_directories(cuampcor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    )
target_compile_definitions(cuampcor PRIVATE
    -D__HIP_PLATFORM_HCC__=1
    -D__HIP__=1
    )
target_link_libraries(cuampcor PRIVATE
    fftw3
    fftw3f
    GDAL::GDAL
    tbb
    )

install(TARGETS cuampcor
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
