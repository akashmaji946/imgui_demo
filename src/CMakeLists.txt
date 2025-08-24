cmake_minimum_required(VERSION 3.10)
project(imgui_demo LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Executable name (same as Makefile)
set(EXE_NAME demo)

# Paths
set(IMGUI_DIR "${CMAKE_CURRENT_LIST_DIR}/..")
set(IMGUI_SRC_DIR "${IMGUI_DIR}/imgui")
set(BACKENDS_DIR "${IMGUI_DIR}/backends")

set(INCLUDE_DIR "${IMGUI_DIR}/include")

# Sources (mirroring the Makefile)
set(SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
    


    ${IMGUI_SRC_DIR}/imgui.cpp
    ${IMGUI_SRC_DIR}/imgui_demo.cpp
    ${IMGUI_SRC_DIR}/imgui_draw.cpp
    ${IMGUI_SRC_DIR}/imgui_tables.cpp
    ${IMGUI_SRC_DIR}/imgui_widgets.cpp
    ${BACKENDS_DIR}/imgui_impl_glfw.cpp
    ${BACKENDS_DIR}/imgui_impl_opengl3.cpp
)

add_executable(${EXE_NAME} ${SOURCES})

target_include_directories(${EXE_NAME} PRIVATE
    ${INCLUDE_DIR}
    ${IMGUI_SRC_DIR}
    ${BACKENDS_DIR}
)

# Platform-specific libs
if(APPLE)
    # macOS frameworks and GLFW
    find_package(OpenGL REQUIRED)
    # Try pkg-config for glfw first (Homebrew provides it)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(GLFW3 QUIET IMPORTED_TARGET glfw3)
    endif()
    if(GLFW3_FOUND)
        target_link_libraries(${EXE_NAME} PRIVATE OpenGL::GL PkgConfig::GLFW3)
    else()
        # Fallback: try CMake package
        find_package(glfw3 REQUIRED)
        target_link_libraries(${EXE_NAME} PRIVATE OpenGL::GL glfw )
    endif()
    # macOS frameworks needed by GLFW
    find_library(COCOA_FRAMEWORK Cocoa REQUIRED)
    find_library(IOKIT_FRAMEWORK IOKit REQUIRED)
    find_library(COREVIDEO_FRAMEWORK CoreVideo REQUIRED)
    target_link_libraries(${EXE_NAME} PRIVATE
        ${COCOA_FRAMEWORK}
        ${IOKIT_FRAMEWORK}
        ${COREVIDEO_FRAMEWORK}
    )
elseif(WIN32)
    # MinGW: link with preinstalled glfw3 and system libs
    find_package(OpenGL REQUIRED)
    target_link_libraries(${EXE_NAME} PRIVATE
        OpenGL::GL
        glfw3
        gdi32
        opengl32
        imm32
    )
else()
    # Linux: use pkg-config for glfw3 and link against GL
    find_package(OpenGL REQUIRED)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GLFW3 REQUIRED IMPORTED_TARGET glfw3)
    target_link_libraries(${EXE_NAME} PRIVATE OpenGL::GL PkgConfig::GLFW3)
endif()

# Warnings similar to -Wall -Wformat in Makefile (when using GCC/Clang)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(${EXE_NAME} PRIVATE -Wall -Wformat)
endif()

# ------------------------------------------------------------
# Optional CUDA ImGui viewer target (cuda_demo)
# ------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    # Try modern CUDAToolkit package
    find_package(CUDAToolkit QUIET)

    set(CUDA_DEMO_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/cuda_imgui_main.cu
        ${IMGUI_SRC_DIR}/imgui.cpp
        ${IMGUI_SRC_DIR}/imgui_demo.cpp
        ${IMGUI_SRC_DIR}/imgui_draw.cpp
        ${IMGUI_SRC_DIR}/imgui_tables.cpp
        ${IMGUI_SRC_DIR}/imgui_widgets.cpp
        ${BACKENDS_DIR}/imgui_impl_glfw.cpp
        ${BACKENDS_DIR}/imgui_impl_opengl3.cpp
    )

    add_executable(cuda_demo ${CUDA_DEMO_SOURCES})
    target_include_directories(cuda_demo PRIVATE
        ${INCLUDE_DIR}
        ${IMGUI_SRC_DIR}
        ${BACKENDS_DIR}
        ${IMGUI_DIR}/common
    )

    # Match platform libs used for demo
    if(APPLE)
        find_package(OpenGL REQUIRED)
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(GLFW3 QUIET IMPORTED_TARGET glfw3)
        endif()
        if(GLFW3_FOUND)
            target_link_libraries(cuda_demo PRIVATE OpenGL::GL PkgConfig::GLFW3)
        else()
            find_package(glfw3 REQUIRED)
            target_link_libraries(cuda_demo PRIVATE OpenGL::GL glfw)
        endif()
        find_library(COCOA_FRAMEWORK Cocoa REQUIRED)
        find_library(IOKIT_FRAMEWORK IOKit REQUIRED)
        find_library(COREVIDEO_FRAMEWORK CoreVideo REQUIRED)
        target_link_libraries(cuda_demo PRIVATE
            ${COCOA_FRAMEWORK}
            ${IOKIT_FRAMEWORK}
            ${COREVIDEO_FRAMEWORK}
        )
    elseif(WIN32)
        find_package(OpenGL REQUIRED)
        target_link_libraries(cuda_demo PRIVATE
            OpenGL::GL
            glfw3
            gdi32
            opengl32
            imm32
        )
    else()
        find_package(OpenGL REQUIRED)
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(GLFW3 REQUIRED IMPORTED_TARGET glfw3)
        target_link_libraries(cuda_demo PRIVATE OpenGL::GL PkgConfig::GLFW3)
    endif()

    # Link CUDA libs (prefer CUDAToolkit imported targets if found)
    if(CUDAToolkit_FOUND)
        target_link_libraries(cuda_demo PRIVATE CUDA::cudart CUDA::curand)
    else()
        # Fallback to legacy names
        find_library(CUDA_CUDART_LIBRARY cudart)
        find_library(CUDA_CURAND_LIBRARY curand)
        if(CUDA_CUDART_LIBRARY)
            target_link_libraries(cuda_demo PRIVATE ${CUDA_CUDART_LIBRARY})
        endif()
        if(CUDA_CURAND_LIBRARY)
            target_link_libraries(cuda_demo PRIVATE ${CUDA_CURAND_LIBRARY})
        endif()
    endif()

    # Compiler options
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(cuda_demo PRIVATE -Wall -Wformat)
    endif()

    # Set CUDA standard
    set_target_properties(cuda_demo PROPERTIES
        CUDA_STANDARD 11
        CUDA_STANDARD_REQUIRED YES
    )
endif()
