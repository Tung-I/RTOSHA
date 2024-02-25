# PROCESS OVERVIEW
# Configuration: CMake processes the CMakeLists.txt file, checking for required packages, 
# setting up variables, and preparing build configurations.
# Generation: CMake generates build files (Makefiles, Visual Studio project files, etc.)
# tailored to your system's build environment.
# Compilation: The build system compiles source files into object files,
# using the appropriate compiler for each file type (C++, CUDA).
# Linking: Object files are linked together with external libraries to create the final executables.
# Installation: Based on the install rules, the build system copies the necessary files to their designated locations.


cmake_minimum_required(VERSION 3.7)
project(Ringmaster)

# # Set the path to LibTorch
# list(APPEND CMAKE_PREFIX_PATH "/home/ubuntu/libtorch")

set(SENDER_SOURCES
 ${CMAKE_CURRENT_SOURCE_DIR}/sender.cpp
)
set(RECEIVER_SOURCES
 ${CMAKE_CURRENT_SOURCE_DIR}/receiver.cpp
)
set(NV_ENC_SOURCES
 ${NV_ENC_DIR}/NvEncoder.cpp
 ${NV_ENC_DIR}/NvEncoderCuda.cpp
 ${NV_DEC_DIR}/NvDecoder.cpp
 )
set(NV_ENC_HDRS
 ${NV_ENC_DIR}/NvEncoder.h
 ${NV_ENC_DIR}/NvEncoderCuda.h
 ${NV_DEC_DIR}/NvDecoder.h
 ${NVCODEC_PUBLIC_INTERFACE_DIR}/nvEncodeAPI.h
 ${NVCODEC_UTILS_DIR}/NvCodecUtils.h
 ${NVCODEC_UTILS_DIR}/NvEncoderCLIOptions.h
 ${NVCODEC_UTILS_DIR}/ColorSpace.h
 ${NVCODEC_UTILS_DIR}/FFmpegStreamer.h
 ${NVCODEC_UTILS_DIR}/FFmpegDemuxer.h
)
set (RM_SOURCES
 ${RM_APP_DIR}/HWEncoder.cc
 ${RM_APP_DIR}/HWDecoder.cc
 ${RM_APP_DIR}/protocol.cc
 ${RM_UTILS_DIR}/address.cc
 ${RM_UTILS_DIR}/conversion.cc
 ${RM_UTILS_DIR}/epoller.cc
 ${RM_UTILS_DIR}/file_descriptor.cc
 ${RM_UTILS_DIR}/mmap.cc
 ${RM_UTILS_DIR}/poller.cc
 ${RM_UTILS_DIR}/serialization.cc
 ${RM_UTILS_DIR}/socket.cc
 ${RM_UTILS_DIR}/split.cc
 ${RM_UTILS_DIR}/timerfd.cc
 ${RM_UTILS_DIR}/timestamp.cc
 ${RM_UTILS_DIR}/udp_socket.cc
 ${RM_VIDEO_DIR}/image.cc
 ${RM_VIDEO_DIR}/sdl.cc
 ${RM_VIDEO_DIR}/v4l2.cc
 ${RM_VIDEO_DIR}/yuv4mpeg.cc
)
set (RM_HDRS
 ${RM_APP_DIR}/HWEncoder.hh
 ${RM_APP_DIR}/HWDecoder.hh
 ${RM_APP_DIR}/protocol.hh
 ${RM_UTILS_DIR}/address.hh
 ${RM_UTILS_DIR}/conversion.hh
 ${RM_UTILS_DIR}/epoller.hh
 ${RM_UTILS_DIR}/exception.hh
 ${RM_UTILS_DIR}/file_descriptor.hh
 ${RM_UTILS_DIR}/mmap.hh
 ${RM_UTILS_DIR}/poller.hh
 ${RM_UTILS_DIR}/serialization.hh
 ${RM_UTILS_DIR}/socket.hh
 ${RM_UTILS_DIR}/split.hh
 ${RM_UTILS_DIR}/timerfd.hh
 ${RM_UTILS_DIR}/timestamp.hh
 ${RM_UTILS_DIR}/udp_socket.hh
 ${RM_VIDEO_DIR}/image.hh
 ${RM_VIDEO_DIR}/sdl.hh
 ${RM_VIDEO_DIR}/v4l2.hh
 ${RM_VIDEO_DIR}/video_input.hh
 ${RM_VIDEO_DIR}/yuv4mpeg.hh
)
set(NV_ENC_CUDA_UTILS
 ${NVCODEC_UTILS_DIR}/ColorSpace.cu
)


# Locates the CUDA toolkit on the system.
# It sets variables like CUDA_INCLUDE_DIRS and CUDA_LIBRARIES
find_package(CUDA) 
# find_package(Torch REQUIRED)

# Configures CUDA compiler
# Adjusts the flags passed to the CUDA NVCC compiler
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_50,code=\"sm_50,compute_50\")
if (WIN32 AND CUDA_VERSION_MAJOR GREATER_EQUAL 11 AND CUDA_VERSION_MINOR GREATER_EQUAL 7)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler /MT)
endif()
if ( CMAKE_COMPILER_IS_GNUCC )
    if(NOT "${CUDA_NVCC_FLAGS}" MATCHES "-std=c\\+\\+11" )
        list(APPEND CUDA_NVCC_FLAGS -std=c++11)
    endif()
endif()

# Creates executables.
# CUDA source files are compiled with NVCC, while C++ sources are compiled with the system C++ compiler.
cuda_add_executable(sender ${SENDER_SOURCES} ${RM_SOURCES} ${NV_ENC_SOURCES} ${NV_ENC_CUDA_UTILS} ${RM_HDRS} ${NV_ENC_HDRS} ${NV_DEC_HDRS} ${NV_FFMPEG_HDRS})
cuda_add_executable(receiver ${RECEIVER_SOURCES} ${RM_SOURCES} ${NV_ENC_SOURCES} ${NV_ENC_CUDA_UTILS} ${RM_HDRS} ${NV_ENC_HDRS} ${NV_DEC_HDRS} ${NV_FFMPEG_HDRS})

# Sets properties for each target.
# This includes enabling CUDA separate compilation to handle CUDA dependencies
# and specifying the C++ standard version.
set_target_properties(sender PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
set_target_properties(receiver PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Specifies directories where the compiler should look for header files during compilation.
# This is essential for resolving #include directives in your source code.
target_include_directories(sender PUBLIC ${CUDA_INCLUDE_DIRS}
 ${NVCODEC_PUBLIC_INTERFACE_DIR}
 ${NVCODEC_UTILS_DIR}
 ${NV_FFMPEG_HDRS}
 ${NV_CODEC_DIR}
 ${RM_UTILS_DIR}
 ${RM_VIDEO_DIR}
)
target_include_directories(receiver PUBLIC ${CUDA_INCLUDE_DIRS}
 ${NVCODEC_PUBLIC_INTERFACE_DIR}
 ${NVCODEC_UTILS_DIR}
 ${NV_FFMPEG_HDRS}
 ${NV_CODEC_DIR}
 ${RM_UTILS_DIR}
 ${RM_VIDEO_DIR}
)


# include_directories(
#     ${TORCH_INCLUDE_DIRS} # Add the LibTorch include directories
# )

# Ensures PkgConfig is available,
# which is a tool to help manage library flags for compilers and linkers.
# pkg_check_modules(...): Uses PkgConfig to find and set flags for external libraries not found directly by CMake's find_package.
find_package(PkgConfig REQUIRED)
pkg_check_modules(VPX REQUIRED IMPORTED_TARGET vpx)
pkg_check_modules(SDL2 REQUIRED IMPORTED_TARGET sdl2)

# Links libraries to your executable targets.
# This process involves resolving symbols (functions, variables) used in your code 
# but defined in external libraries (like CUDA runtime libraries, libavcodec for FFmpeg, or SDL2).
target_link_libraries(sender ${CUDA_CUDA_LIBRARY} ${CMAKE_DL_LIBS} ${NVENCODEAPI_LIB} ${CUVID_LIB} ${AVCODEC_LIB}
 ${AVFORMAT_LIB} ${AVUTIL_LIB} ${SWRESAMPLE_LIB} PkgConfig::VPX PkgConfig::SDL2)
target_link_libraries(receiver ${CUDA_CUDA_LIBRARY} ${CMAKE_DL_LIBS} ${NVENCODEAPI_LIB} ${CUVID_LIB} ${AVCODEC_LIB}
${AVFORMAT_LIB} ${AVUTIL_LIB} ${SWRESAMPLE_LIB} PkgConfig::VPX PkgConfig::SDL2)

# add_executable(sender ${SENDER_SOURCES} ${ADDITIONAL_SOURCES})
# target_link_libraries(sender 
#     ${TORCH_LIBRARIES} # Link against LibTorch
#     # Link other libraries as needed
# )
# The ${TORCH_LIBRARIES} variable is automatically defined by the find_package(Torch REQUIRED)

 # Specifies rules for installing your targets (executables, libraries).
 # This determines where files go when you run make install
install(TARGETS sender RUNTIME DESTINATION ${NVCODEC_SAMPLES_INSTALL_DIR})
install(TARGETS receiver RUNTIME DESTINATION ${NVCODEC_SAMPLES_INSTALL_DIR})
