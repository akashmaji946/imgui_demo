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
static Point camera_lookat = Point(0, 0, -1);
static Point camera_lookfrom = Point(0, 0, 0);
static bool camera_changed = true;


// ==== Global camera state ====
static float mouse_x = 0.0f;    // yaw (start at 0 for natural orientation)
static float mouse_y = 20.0f;   // pitch (slightly above horizontal)
static float camera_radius = 10.0f; // closer default zoom

// ==== Mouse interaction globals ====
static bool left_mouse_pressed = false;
static double last_mouse_x = 0.0, last_mouse_y = 0.0;

// ==== Callbacks ====
void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (left_mouse_pressed) {
        float dx = static_cast<float>(xpos - last_mouse_x);
        float dy = static_cast<float>(ypos - last_mouse_y);

        float sensitivity = 0.1f;   // smoother, slower
        mouse_x -= dx * sensitivity; // invert X for natural orbit
        mouse_y -= dy * sensitivity; // invert Y for natural orbit

        // clamp pitch
        if (mouse_y > 89.0f) mouse_y = 89.0f;
        if (mouse_y < -89.0f) mouse_y = -89.0f;

        camera_changed = true;
    }
    last_mouse_x = xpos;
    last_mouse_y = ypos;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            left_mouse_pressed = true;
            glfwGetCursorPos(window, &last_mouse_x, &last_mouse_y);
        } else if (action == GLFW_RELEASE) {
            left_mouse_pressed = false;
        }
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    camera_radius -= (float)yoffset * 0.1f; // faster zoom

    if (camera_radius < 0.5f) camera_radius = 0.5f; 
    if (camera_radius > 20.0f) camera_radius = 20.0f;
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

    // Enable mouse callbacks
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

    OFFModel* model;
    if(argv[1] != NULL) {
        model = readOffFile(argv[1]);
        if(model == NULL) {
            std::cerr << "Failed to read OFF file: " << argv[1] << std::endl;
            return 1;
        }
    } else {
        model = NULL;
    }

    // Compute bounding box
    Point min_pt(1e9, 1e9, 1e9);
    Point max_pt(-1e9, -1e9, -1e9);

    for (int i = 0; i < model->numberOfVertices; i++) {
        Vector v(model->vertices[i].x, model->vertices[i].y, model->vertices[i].z);
        min_pt.m_x = std::min(min_pt.m_x, v.m_x);
        min_pt.m_y = std::min(min_pt.m_y, v.m_y);
        min_pt.m_z = std::min(min_pt.m_z, v.m_z);

        max_pt.m_x = std::max(max_pt.m_x, v.m_x);
        max_pt.m_y = std::max(max_pt.m_y, v.m_y);
        max_pt.m_z = std::max(max_pt.m_z, v.m_z);
    }

    // Center of model
    camera_lookat = Point(
        0.5 * (min_pt.m_x + max_pt.m_x),
        0.5 * (min_pt.m_y + max_pt.m_y),
        0.5 * (min_pt.m_z + max_pt.m_z)
    );

    // Estimate radius so object fits nicely
    double dx = max_pt.m_x - min_pt.m_x;
    double dy = max_pt.m_y - min_pt.m_y;
    double dz = max_pt.m_z - min_pt.m_z;
    double max_extent = std::max({dx, dy, dz});
    camera_radius = (float)(max_extent * 1.5);  // distance from center


    Renderer renderer(image_width, image_height, samples_per_pixel, model);



    // Initial render setup
    renderer.update_camera(camera_lookfrom, camera_lookat, camera_vfov);
    renderer.render_frame(kernel_choice);

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
            double pitch_rad = mouse_y * M_PI / 180.0;
            double yaw_rad   = mouse_x * M_PI / 180.0;

            camera_lookfrom.m_x = camera_lookat.m_x + camera_radius * cos(yaw_rad) * cos(pitch_rad);
            camera_lookfrom.m_y = camera_lookat.m_y + camera_radius * sin(pitch_rad);
            camera_lookfrom.m_z = camera_lookat.m_z + camera_radius * sin(yaw_rad) * cos(pitch_rad);

            renderer.update_camera(camera_lookfrom, camera_lookat, camera_vfov);
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
        changed_local |= ImGui::SliderFloat("Yaw", &mouse_x, -360.0f, 360.0f);
        changed_local |= ImGui::SliderFloat("Pitch", &mouse_y, -89.0f, 89.0f);
        changed_local |= ImGui::SliderFloat("Zoom", &camera_radius, -1000.0f, 1000.0f);
        changed_local |= ImGui::SliderInt("Samples Per Pixel", &samples_per_pixel, 1, 100);
        changed_local |= ImGui::SliderInt("Width", &image_width, 160, 1920);
        changed_local |= ImGui::SliderInt("Height", &image_height, 90, 1080);
        changed_local |= ImGui::SliderFloat("FOV", (float*)&camera_vfov, 5.0f, 60.0f);
        if (ImGui::Combo("Kernel", &kernel_choice, "simple\0antialias\0cool\0")) changed_local = true;

        if (changed_local) {
            camera_changed = true;
            renderer.resize(image_width, image_height, samples_per_pixel);
            renderer.update_camera(camera_lookfrom, camera_lookat, camera_vfov);
            renderer.render_frame(kernel_choice);
        }

        if (ImGui::Button("Renders")) {
            renderer.resize(image_width, image_height, samples_per_pixel);
            renderer.update_camera(camera_lookfrom, camera_lookat, camera_vfov);
            renderer.render_frame(kernel_choice);
        }
        ImGui::SameLine();
        if (ImGui::Button("Save PNG")) {
            stbi_write_png("image.png", image_width, image_height, 3,
                           renderer.get_host_image(), image_width * 3);
        }

        if(ImGui::Button("Reset")) {
            image_width = 1080;
            image_height = 720;
            samples_per_pixel = 1;
            kernel_choice = 2;  
            camera_vfov = 20.0;
            camera_lookat = Point(
                0.5 * (min_pt.m_x + max_pt.m_x),
                0.5 * (min_pt.m_y + max_pt.m_y),
                0.5 * (min_pt.m_z + max_pt.m_z)
            );
            camera_radius = (float)(max_extent * 1.5);
            mouse_x = 0.0f;
            mouse_y = 20.0f;
            
            renderer.resize(image_width, image_height, samples_per_pixel);
            renderer.update_camera(camera_lookfrom, camera_lookat, camera_vfov);
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
            quad_width = quad_height * image_aspect;
            x_offset = (dw - quad_width) * 0.5f;
            y_offset = 0;
        } else {
            quad_width = (float)dw;
            quad_height = quad_width / image_aspect;
            x_offset = 0;
            y_offset = (dh - quad_height) * 0.5f;
        }

        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(x_offset, y_offset + quad_height);
        glTexCoord2f(1.0f, 1.0f); glVertex2f(x_offset + quad_width, y_offset + quad_height);
        glTexCoord2f(1.0f, 0.0f); glVertex2f(x_offset + quad_width, y_offset);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(x_offset, y_offset);
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
