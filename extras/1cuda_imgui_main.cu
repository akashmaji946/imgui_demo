#include "../common/common.hpp"
// ImGui + GLFW + OpenGL3
#include "../imgui/imgui.h"
#include "../backends/imgui_impl_glfw.h"
#include "../backends/imgui_impl_opengl3.h"

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

// Forward declare your kernels from the original snippet
__global__ void render(unsigned char* image, const Camera* cam, const Scene* scene, int image_width, int image_height);
__global__ void render_antialias(unsigned char* image, const Camera* cam, const Scene* scene, int image_width, int image_height);
__global__ void render_cool(unsigned char* image, const Camera* cam, const Scene* scene, int image_width, int image_height);

static void glfw_error_callback(int error, const char* description) {
    std::fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

// Global camera state
static double camera_vfov = 20.0;
static Point camera_lookfrom = Point(0, 0, 0);
static Point camera_lookat = Point(0, 0, -1);
static bool camera_changed = true;

// Mouse tracking
static double last_x = 0.0;
static double last_y = 0.0;
static bool first_mouse = true;
static bool moving = false;
static float yaw = 0.0f;
static float pitch = 0.0f;

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (!moving) return;

    if (first_mouse) {
        last_x = xpos;
        last_y = ypos;
        first_mouse = false;
    }

    double xoffset = xpos - last_x;
    double yoffset = last_y - ypos; 
    last_x = xpos;
    last_y = ypos;

    double sensitivity = 0.1;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw += xoffset;
    pitch += yoffset;

    // Constrain pitch to avoid camera flipping
    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;

    Vector direction;
    direction.m_x = cos(yaw * M_PI / 180.0) * cos(pitch * M_PI / 180.0);
    direction.m_y = sin(pitch * M_PI / 180.0);
    direction.m_z = sin(yaw * M_PI / 180.0) * cos(pitch * M_PI / 180.0);
    
    camera_lookat = camera_lookfrom + direction;
    camera_changed = true;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && !ImGui::IsAnyItemActive()) {
        if (action == GLFW_PRESS) {
            moving = true;
            first_mouse = true;
        } else if (action == GLFW_RELEASE) {
            moving = false;
        }
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    if (ImGui::IsAnyItemActive()) return;

    double zoom_factor = 0.01;
    Vector move_dir = camera_lookat - camera_lookfrom;
    camera_lookfrom = camera_lookfrom + move_dir * yoffset * zoom_factor;

    camera_changed = true;
}

int main(int, char**) {
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

    GLFWwindow* window = glfwCreateWindow(1280, 800, "CUDA ImGui Viewer", nullptr, nullptr);
    if (!window) return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // Register callbacks for camera movement
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
    int image_width = 1080;
    int image_height = 720;
    int samples_per_pixel = 100;
    int kernel_choice = 2;

    Renderer renderer(image_width, image_height, samples_per_pixel, model);
    
    // Initial render
    renderer.update_camera(camera_lookfrom, camera_lookat, camera_vfov);
    renderer.render_frame(kernel_choice);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Check for camera movement and update renderer
        if (camera_changed) {
            renderer.update_camera(camera_lookfrom, camera_lookat, camera_vfov);
            camera_changed = false;
            renderer.render_frame(kernel_choice);
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Controls
        ImGui::Begin("Controls");
        ImGui::Text("CUDA ImGui Viewer");
        ImGui::Separator();
        bool changed = false;
        changed |= ImGui::SliderInt("Samples Per Pixel", &samples_per_pixel, 1, 500);
        changed |= ImGui::SliderInt("Width", &image_width, 160, 1920);
        changed |= ImGui::SliderInt("Height", &image_height, 90, 1080);
        changed |= ImGui::SliderFloat("FOV", (float*)&camera_vfov, 5.0f, 60.0f);
        if (ImGui::Combo("Kernel", &kernel_choice, "simple\0antialias\0cool\0")) changed = true;
        if (changed) {
            renderer.resize(image_width, image_height, samples_per_pixel);
            renderer.render_frame(kernel_choice);
        }
        if (ImGui::Button("Render")) {
            renderer.render_frame(kernel_choice);
        }
        ImGui::SameLine();
        if (ImGui::Button("Save PNG")) {
            stbi_write_png("image.png", image_width, image_height, 3, renderer.get_host_image(), image_width * 3);
        }
        ImGui::End();

        // Render to the main window
        ImGui::Render();
        int dw, dh; glfwGetFramebufferSize(window, &dw, &dh);
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
        
        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(0, 0);
        glTexCoord2f(1.0f, 1.0f); glVertex2f(dw, 0);
        glTexCoord2f(1.0f, 0.0f); glVertex2f(dw, dh);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(0, dh);
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