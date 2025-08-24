#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>

// ImGui + GLFW + OpenGL3
#include "../imgui/imgui.h"
#include "../backends/imgui_impl_glfw.h"
#include "../backends/imgui_impl_opengl3.h"
#include "../common/OFFReader.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../common/stb_image_write.h"

#define GL_SILENCE_DEPRECATION
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#else
#include <GL/gl.h>
#endif
#include <GLFW/glfw3.h>

// Project includes
#include "../include/Renderer.hpp"
#include "../include/Point.hpp"
#include "../include/Vector.hpp"
#include "../include/Camera.hpp"
#include "../include/Scene.hpp"

// Forward declare kernels
__global__ void render(unsigned char* image, const Camera* cam, const Scene* scene, int image_width, int image_height);
__global__ void render_antialias(unsigned char* image, const Camera* cam, const Scene* scene, int image_width, int image_height);
__global__ void render_cool(unsigned char* image, const Camera* cam, const Scene* scene, int image_width, int image_height);

static void glfw_error_callback(int error, const char* description) {
    std::fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

// ==== Global camera state ====
static double camera_vfov = 20.0;
static Point  camera_lookat = Point(0, 0, -1);     // target (model center)
static Point  camera_lookfrom = Point(0, 0, 0);
static Vector camera_up = Vector(0, 1, 0);
static bool   camera_changed = true;

// Yaw/Pitch orbit (degrees) + radius
static float yaw_deg   = 0.0f;
static float pitch_deg = 20.0f;
static float camera_radius = 3.0f;

// Zoom bounds that adapt to model size
static float min_radius = 0.25f;
static float max_radius = 50.0f;

// ==== Mouse interaction globals ====
static bool   left_mouse_pressed = false;
static double last_mouse_x = 0.0, last_mouse_y = 0.0;

// ---- Helpers ----
static inline void update_camera_from_spherical() {
    // Convert spherical (radius, yaw, pitch) around camera_lookat -> camera_lookfrom
    const double pitch_rad = pitch_deg * M_PI / 180.0;
    const double yaw_rad   = yaw_deg   * M_PI / 180.0;

    camera_lookfrom.m_x = camera_lookat.m_x + camera_radius * std::cos(yaw_rad) * std::cos(pitch_rad);
    camera_lookfrom.m_y = camera_lookat.m_y + camera_radius * std::sin(pitch_rad);
    camera_lookfrom.m_z = camera_lookat.m_z + camera_radius * std::sin(yaw_rad) * std::cos(pitch_rad);
}

// Utility: set center/radius from model AABB
static inline void fit_to_bounds(const Point& min_pt, const Point& max_pt) {
    camera_lookat = Point(
        0.5 * (min_pt.m_x + max_pt.m_x),
        0.5 * (min_pt.m_y + max_pt.m_y),
        0.5 * (min_pt.m_z + max_pt.m_z)
    );

    const double dx = max_pt.m_x - min_pt.m_x;
    const double dy = max_pt.m_y - min_pt.m_y;
    const double dz = max_pt.m_z - min_pt.m_z;
    const double max_extent = std::max({dx, dy, dz, 1e-6}); // avoid zero

    // Set radius & bounds relative to model size (stable for tiny/huge models)
    camera_radius = static_cast<float>(max_extent * 1.5);
    min_radius    = static_cast<float>(max_extent * 0.05);
    max_radius    = static_cast<float>(max_extent * 10.0);

    // Sensible initial angles
    yaw_deg = 0.0f;
    pitch_deg = 20.0f;

    update_camera_from_spherical();
    camera_changed = true;
}

// ==== Callbacks ====
void mouse_callback(GLFWwindow* /*window*/, double xpos, double ypos) {
    if (left_mouse_pressed) {
        float dx = static_cast<float>(xpos - last_mouse_x);
        float dy = static_cast<float>(ypos - last_mouse_y);

        // Natural orbit: drag right -> rotate right; drag up -> look up
        const float sensitivity = 0.35f; // deg per pixel (snappy but smooth)
        yaw_deg   += dx * sensitivity;
        pitch_deg += dy * sensitivity;

        // Clamp pitch to avoid flipping
        if (pitch_deg > 89.0f)  pitch_deg = 89.0f;
        if (pitch_deg < -89.0f) pitch_deg = -89.0f;

        update_camera_from_spherical();
        camera_changed = true;
    }
    last_mouse_x = xpos;
    last_mouse_y = ypos;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int /*mods*/) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            left_mouse_pressed = true;
            glfwGetCursorPos(window, &last_mouse_x, &last_mouse_y);
        } else if (action == GLFW_RELEASE) {
            left_mouse_pressed = false;
        }
    }
}

void scroll_callback(GLFWwindow* /*window*/, double /*xoffset*/, double yoffset) {
    // Multiplicative zoom feels better than additive
    float scale = 1.0f - static_cast<float>(yoffset) * 0.1f; // wheel notch ≈ 10%
    if (scale < 0.1f) scale = 0.1f;
    if (scale > 2.0f) scale = 2.0f;

    camera_radius *= scale;
    if (camera_radius < min_radius) camera_radius = min_radius;
    if (camera_radius > max_radius) camera_radius = max_radius;

    update_camera_from_spherical();
    camera_changed = true;
}

int main(int argc, char** argv) {
    // Window + GL
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) return 1;

#if defined(IMGUI_IMPL_OPENGL_ES2)
    const char* glsl_version = "#version 100";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#else
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
#endif

    GLFWwindow* window = glfwCreateWindow(1920, 1080, "CUDA ImGui Viewer", nullptr, nullptr);
    if (!window) return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // Mouse callbacks
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // ImGui setup
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Renderer setup
    int image_width = 640;
    int image_height = 480;
    int samples_per_pixel = 1;
    int kernel_choice = 2;

    // Compute bounding box for centering & zoom bounds
    Point min_pt( 1e9,  1e9,  1e9);
    Point max_pt(-1e9, -1e9, -1e9);

    // Load model (required for centering)
    OFFModel* model = nullptr;
    if (argc >= 2 && argv[1] != nullptr) {
        model = readOffFile(argv[1]);
        if (model == nullptr) {
            std::cerr << "Failed to read OFF file: " << argv[1] << std::endl;
            return 1;
        }

        for (int i = 0; i < model->numberOfVertices; ++i) {
            const Vector v(model->vertices[i].x, model->vertices[i].y, model->vertices[i].z);
            min_pt.m_x = std::min(min_pt.m_x, v.m_x);
            min_pt.m_y = std::min(min_pt.m_y, v.m_y);
            min_pt.m_z = std::min(min_pt.m_z, v.m_z);
            max_pt.m_x = std::max(max_pt.m_x, v.m_x);
            max_pt.m_y = std::max(max_pt.m_y, v.m_y);
            max_pt.m_z = std::max(max_pt.m_z, v.m_z);
        }

        // Center & fit (also computes initial camera_lookfrom)
        fit_to_bounds(min_pt, max_pt);



    } else {
        std::cerr << "Usage: " << argv[0] << " <model.off>\n";
        std::cerr << "No model file provided, using demo model.\n";
        fit_to_bounds(Point(-10, -10, -10), Point(10, 10, 10));
        //fit_to_bounds(Point(-1, -1, -1, -1), Point(1, 1, 1));
    }

    Renderer renderer(image_width, image_height, samples_per_pixel, model);

    // Initial render setup — use the computed camera_lookfrom/lookat
    renderer.update_camera(camera_lookfrom, camera_lookat, camera_up,camera_vfov);
    renderer.render_frame(kernel_choice);
    camera_changed = false;

    // FPS tracking
    double last_time = glfwGetTime();
    int frame_count = 0;
    double fps = 0.0;

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // FPS calculation
        double current_time = glfwGetTime();
        frame_count++;
        if (current_time - last_time >= 1.0) {
            fps = static_cast<double>(frame_count) / (current_time - last_time);
            frame_count = 0;
            last_time = current_time;
        }

        // Update camera if changed
        if (camera_changed) {
            renderer.update_camera(camera_lookfrom, camera_lookat, camera_up, camera_vfov);
            renderer.render_frame(kernel_choice);
            camera_changed = false;
        }

        // ImGui
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Controls");
        ImGui::Text("CUDA ImGui Viewer");
        ImGui::Separator();
        ImGui::Text("FPS: %.2f", fps);
        ImGui::Separator();

        bool changed_local = false;

        // Rotation controls (degrees)
        changed_local |= ImGui::SliderFloat("Yaw (deg)",   &yaw_deg,   -720.0f, 720.0f);
        changed_local |= ImGui::SliderFloat("Pitch (deg)", &pitch_deg,  -89.0f,  89.0f);

        // When sliders move, recompute camera and mark changed
        if (changed_local) {
            update_camera_from_spherical();
        }

        // Logarithmic zoom slider (model-aware bounds)
        ImGui::PushItemWidth(260.0f);
        ImGuiSliderFlags zoomFlags = ImGuiSliderFlags_Logarithmic;
        bool zoom_changed = ImGui::SliderFloat("Zoom (radius)", &camera_radius, min_radius, max_radius, "%.3f", zoomFlags);
        ImGui::PopItemWidth();

        if (zoom_changed) {
            update_camera_from_spherical();
            changed_local = true;
        }

        // Other render controls
        changed_local |= ImGui::SliderInt("Samples Per Pixel", &samples_per_pixel, 1, 100);
        changed_local |= ImGui::SliderInt("Width",  &image_width, 160, 1920);
        changed_local |= ImGui::SliderInt("Height", &image_height, 90,  1080);
        changed_local |= ImGui::SliderFloat("FOV", (float*)&camera_vfov, 5.0f, 60.0f);
        if (ImGui::Combo("Kernel", &kernel_choice, "simple\0antialias\0cool\0")) changed_local = true;

        if (ImGui::Button("Fit / Center")) {
            fit_to_bounds(min_pt, max_pt);
            changed_local = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Save PNG")) {
            stbi_write_png("image.png", image_width, image_height, 3,
                           renderer.get_host_image(), image_width * 3);
        }

        if (changed_local) {
            camera_changed = true;
            renderer.resize(image_width, image_height, samples_per_pixel);
            // camera updated above already when sliders changed
            renderer.update_camera(camera_lookfrom, camera_lookat, camera_up, camera_vfov);
            renderer.render_frame(kernel_choice);
            camera_changed = false;
        }

        ImGui::Separator();
        if (ImGui::Button("Reset View")) {
            fit_to_bounds(min_pt, max_pt);
            camera_vfov = 20.0;
            renderer.update_camera(camera_lookfrom, camera_lookat, camera_up, camera_vfov);
            renderer.render_frame(kernel_choice);
        }

        ImGui::End();

        if (renderer.is_ready()) {
            renderer.update_texture();
        }

        // Render to window
        ImGui::Render();
        int dw, dh;
        glfwGetFramebufferSize(window, &dw, &dh);
        glViewport(0, 0, dw, dh);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, dw, dh, 0, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, renderer.get_texture_id());

        float window_aspect = (float)dw / dh;
        float image_aspect  = (float)image_width / image_height;
        float quad_width, quad_height;
        float x_offset, y_offset;

        if (window_aspect > image_aspect) {
            quad_height = (float)dh;
            quad_width  = quad_height * image_aspect;
            x_offset = (dw - quad_width) * 0.5f;
            y_offset = 0;
        } else {
            quad_width  = (float)dw;
            quad_height = quad_width / image_aspect;
            x_offset = 0;
            y_offset = (dh - quad_height) * 0.5f;
        }

        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(x_offset,              y_offset + quad_height);
        glTexCoord2f(1.0f, 1.0f); glVertex2f(x_offset + quad_width, y_offset + quad_height);
        glTexCoord2f(1.0f, 0.0f); glVertex2f(x_offset + quad_width, y_offset);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(x_offset,              y_offset);
        glEnd();

        glDisable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 0);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

