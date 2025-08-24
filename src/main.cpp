#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

// ---------- Camera Globals ----------
static float yaw   = 0.0f;    // left-right rotation
static float pitch = 20.0f;   // up-down rotation
static float radius = 3.0f;   // zoom distance
static glm::vec3 target(0.0f, 0.0f, 0.0f); // look-at point

static glm::vec3 cameraPos;
static glm::vec3 cameraFront;
static glm::vec3 cameraUp(0.0f, 1.0f, 0.0f);

static bool leftMousePressed = false;

// ---------- Mouse Input ----------
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) leftMousePressed = true;
        else if (action == GLFW_RELEASE) leftMousePressed = false;
    }
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    static double lastX = xpos, lastY = ypos;
    if (leftMousePressed) {
        float dx = float(xpos - lastX);
        float dy = float(ypos - lastY);

        float sensitivity = 0.3f;
        yaw   += dx * sensitivity;
        pitch -= dy * sensitivity;

        if (pitch > 89.0f) pitch = 89.0f;
        if (pitch < -89.0f) pitch = -89.0f;
    }
    lastX = xpos;
    lastY = ypos;
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    radius -= (float)yoffset * 0.3f;
    if (radius < 0.5f) radius = 0.5f;
    if (radius > 100.0f) radius = 100.0f;
}

// ---------- Update Camera ----------
void updateCamera() {
    float radYaw   = glm::radians(yaw);
    float radPitch = glm::radians(pitch);

    float x = radius * cos(radPitch) * cos(radYaw);
    float y = radius * sin(radPitch);
    float z = radius * cos(radPitch) * sin(radYaw);

    cameraPos = target + glm::vec3(x, y, z);
    cameraFront = glm::normalize(target - cameraPos);
}

// ---------- Render ----------
void renderCube() {
    // Simple cube for demo
    glBegin(GL_QUADS);
    glColor3f(1, 0, 0); glVertex3f(-1, -1, -1); glVertex3f(-1, -1,  1); glVertex3f(-1,  1,  1); glVertex3f(-1,  1, -1);
    glColor3f(0, 1, 0); glVertex3f( 1, -1, -1); glVertex3f( 1, -1,  1); glVertex3f( 1,  1,  1); glVertex3f( 1,  1, -1);
    glColor3f(0, 0, 1); glVertex3f(-1, -1, -1); glVertex3f( 1, -1, -1); glVertex3f( 1, -1,  1); glVertex3f(-1, -1,  1);
    glColor3f(1, 1, 0); glVertex3f(-1,  1, -1); glVertex3f( 1,  1, -1); glVertex3f( 1,  1,  1); glVertex3f(-1,  1,  1);
    glColor3f(0, 1, 1); glVertex3f(-1, -1, -1); glVertex3f( 1, -1, -1); glVertex3f( 1,  1, -1); glVertex3f(-1,  1, -1);
    glColor3f(1, 0, 1); glVertex3f(-1, -1,  1); glVertex3f( 1, -1,  1); glVertex3f( 1,  1,  1); glVertex3f(-1,  1,  1);
    glEnd();
}

// ---------- Main ----------
int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n";
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(800, 600, "Orbit Camera Viewer", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glewInit();

    glEnable(GL_DEPTH_TEST);

    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // Example model bounding box: cube [-1,1]
    glm::vec3 minPt(-1, -1, -1), maxPt(1, 1, 1);
    target = (minPt + maxPt) * 0.5f;
    float maxExtent = std::max({maxPt.x - minPt.x, maxPt.y - minPt.y, maxPt.z - minPt.z});
    radius = maxExtent * 2.0f;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        updateCamera();

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        float aspect = float(width) / float(height);

        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
        glm::mat4 view = glm::lookAt(cameraPos, target, cameraUp);

        glMatrixMode(GL_PROJECTION);
        glLoadMatrixf(glm::value_ptr(projection));
        glMatrixMode(GL_MODELVIEW);
        glLoadMatrixf(glm::value_ptr(view));

        renderCube();

        glfwSwapBuffers(window);
    }

    glfwTerminate();
    return 0;
}
