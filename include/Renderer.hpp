#ifndef RENDERER_H
#define RENDERER_H

#include <GL/gl.h>
#include "../common/common.hpp"
#include "../common/OFFReader.h"
#include "../include/Scene.hpp"
#include "../include/CudaScene.hpp"
#include "../include/Camera.hpp"
#include "../include/Color.hpp"
#include "../include/Ray.hpp"
#include "../include/Triangle.hpp"
#include "../include/Sphere.hpp"
#include "../include/SurfaceInteraction.hpp"
#include "Vector.hpp"

// Forward declare your kernels from the original snippet
__global__ void render(unsigned char* image, const Camera* cam, const Scene* scene, int image_width, int image_height);
__global__ void render_antialias(unsigned char* image, const Camera* cam, const Scene* scene, int image_width, int image_height);
__global__ void render_cool(unsigned char* image, const Camera* cam, const Scene* scene, int image_width, int image_height);


__global__ void render(unsigned char* image, const Camera* cam, const Scene* scene, int image_width, int image_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= image_width || y >= image_height) return;
    Ray r = cam->get_ray(x, y);
    float closest_t = 1e30f; int hit_type = -1; Color pixel_color(0,0,0); SurfaceInteraction closest_si;
    for (int i = 0; i < scene->num_objects; ++i) {
        SurfaceInteraction si;
        if (scene->objects[i].intersect_si(r, si) && si.t < closest_t) {
            closest_t = si.t; hit_type = (scene->objects[i].type == OBJ_TRIANGLE) ? 0 : 1; closest_si = si;
        }
    }
    if (hit_type == 0) pixel_color = Color(1,0,0);
    else if (hit_type == 1) pixel_color = Color(0,0,1);
    else pixel_color = Color(0,0,0);
    int idx = (y * image_width + x) * 3;
    image[idx+0] = static_cast<unsigned char>(255.99f * pixel_color.m_r);
    image[idx+1] = static_cast<unsigned char>(255.99f * pixel_color.m_g);
    image[idx+2] = static_cast<unsigned char>(255.99f * pixel_color.m_b);
}

__global__ void render_antialias(unsigned char* image, const Camera* cam, const Scene* scene, int image_width, int image_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= image_width || y >= image_height) return;
    curandState_t state; int idx = (y * image_width + x) * 3; curand_init(idx, 0, 0, &state);
    Color pixel_color(0,0,0);
    const int num_samples = cam->samples_per_pixel;
    for (int s = 0; s < num_samples; ++s) {
        float rx = curand_uniform(&state) - 0.5f; float ry = curand_uniform(&state) - 0.5f;
        float u = (x + rx); float v = (y + ry);
        Ray r = cam->get_ray(u, v);
        float closest_t = 1e30f; int hit_type = -1; SurfaceInteraction closest_si;
        for (int i = 0; i < scene->num_objects; ++i) {
            SurfaceInteraction si;
            if (scene->objects[i].intersect_si(r, si) && si.t < closest_t) {
                closest_t = si.t; hit_type = (scene->objects[i].type == OBJ_TRIANGLE) ? 0 : 1; closest_si = si;
            }
        }
        if (hit_type == 0) pixel_color += Color(1,0,0);
        else if (hit_type == 1) pixel_color += Color(0,0,1);
    }
    pixel_color /= static_cast<float>(num_samples);
    image[idx+0] = static_cast<unsigned char>(255.99f * pixel_color.m_r);
    image[idx+1] = static_cast<unsigned char>(255.99f * pixel_color.m_g);
    image[idx+2] = static_cast<unsigned char>(255.99f * pixel_color.m_b);
}

__global__ void render_cool(unsigned char* image, const Camera* cam, const Scene* scene, int image_width, int image_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= image_width || y >= image_height) return;
    curandState_t state; curand_init(x + y * image_width, 0, 0, &state);
    Color pixel_color(0,0,0); 
    const int num_samples = cam->samples_per_pixel;
    for (int s = 0; s < num_samples; ++s) {
        float rx = curand_uniform(&state) - 0.5f; float ry = curand_uniform(&state) - 0.5f;
        float u = (x + rx); float v = (y + ry);
        Ray r = cam->get_ray(u, v);
        float closest_t = 1e30f; int hit_type = -1; SurfaceInteraction closest_si;
        for (int i = 0; i < scene->num_objects; ++i) {
            SurfaceInteraction si;
            if (scene->objects[i].intersect_si(r, si) && si.t < closest_t) {
                closest_t = si.t; hit_type = (scene->objects[i].type == OBJ_TRIANGLE) ? 0 : 1; closest_si = si;
            }
        }
        if (hit_type != -1) {
            Vector N = (closest_si.normal).unit_vector();
            pixel_color += Color(N.m_x + 1, N.m_y + 1, N.m_z + 1) * 0.5f;
        } else {
            pixel_color += Color(0.1f, 0.1f, 0.1f);
        }
    }
    pixel_color /= static_cast<float>(num_samples);
    pixel_color.m_r = fminf(pixel_color.m_r, 1.0f);
    pixel_color.m_g = fminf(pixel_color.m_g, 1.0f);
    pixel_color.m_b = fminf(pixel_color.m_b, 1.0f);
    int idx = (y * image_width + x) * 3;
    image[idx+0] = static_cast<unsigned char>(255.99f * pixel_color.m_r);
    image[idx+1] = static_cast<unsigned char>(255.99f * pixel_color.m_g);
    image[idx+2] = static_cast<unsigned char>(255.99f * pixel_color.m_b);
}

class Renderer {
public:
    Renderer(int initial_width, int initial_height, int samples_per_pixel, OFFModel* model);
    ~Renderer();

    void render_frame(int kernel_choice);
    void resize(int new_width, int new_height, int samples_per_pixel);
    void update_camera(Point lookfrom, Point lookat, Vector cameraup, double vfov);
    GLuint get_texture_id() const { return tex_id; }
    unsigned char* get_host_image() { return h_image.data(); }
    
    // New methods
    bool is_ready();
    void update_texture();


private:
    void create_texture();
    void destroy_buffers();

    int image_width;
    int image_height;
    int samples_per_pixel;
    
    GLuint tex_id;
    unsigned char* d_image = nullptr;
    std::vector<unsigned char> h_image;
    cudaEvent_t render_complete_event;

    CudaScene cuda_scene;
    OFFModel* model;
};

// --- Renderer Class Implementation ---

Renderer::Renderer(int initial_width, int initial_height, int spp, OFFModel* model)
    : image_width(initial_width), image_height(initial_height), samples_per_pixel(spp), model(model)
    {
    
    size_t img_bytes = static_cast<size_t>(image_width) * image_height * 3;
    h_image.resize(img_bytes);
    checkCudaErrors(cudaMalloc((void**)&d_image, img_bytes));
    
    if(model != NULL) {
        cuda_scene.init(image_width, image_height, samples_per_pixel, model);
    } else {
        cuda_scene.init(image_width, image_height, samples_per_pixel);
    }
    create_texture();
    checkCudaErrors(cudaEventCreate(&render_complete_event));
}

Renderer::~Renderer() {
    destroy_buffers();
    checkCudaErrors(cudaEventDestroy(render_complete_event));
}

void Renderer::destroy_buffers() {
    if (d_image) cudaFree(d_image);
    cuda_scene.destroy();
    glDeleteTextures(1, &tex_id);
}

void Renderer::create_texture() {
    glGenTextures(1, &tex_id);
    glBindTexture(GL_TEXTURE_2D, tex_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, image_width, image_height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Renderer::render_frame(int kernel_choice) {
    dim3 block(16, 16);
    dim3 grid((image_width + block.x - 1) / block.x,
              (image_height + block.y - 1) / block.y);

    switch (kernel_choice) {
        case 0: render<<<grid, block>>>(d_image, cuda_scene.d_cam, cuda_scene.d_scene, image_width, image_height); break;
        case 1: render_antialias<<<grid, block>>>(d_image, cuda_scene.d_cam, cuda_scene.d_scene, image_width, image_height); break;
        default: render_cool<<<grid, block>>>(d_image, cuda_scene.d_cam, cuda_scene.d_scene, image_width, image_height); break;
    }
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaEventRecord(render_complete_event, 0));
}

bool Renderer::is_ready() {
    return cudaEventQuery(render_complete_event) == cudaSuccess;
}

void Renderer::update_texture() {
    checkCudaErrors(cudaMemcpy(h_image.data(), d_image, image_width * image_height * 3, cudaMemcpyDeviceToHost));
    
    glBindTexture(GL_TEXTURE_2D, tex_id);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, GL_RGB, GL_UNSIGNED_BYTE, h_image.data());
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Renderer::resize(int new_width, int new_height, int spp) {
    // Wait for any previous render to finish before resizing
    checkCudaErrors(cudaEventSynchronize(render_complete_event));

    image_width = new_width;
    image_height = new_height;
    samples_per_pixel = spp;
    size_t new_bytes = static_cast<size_t>(image_width) * image_height * 3;
    h_image.assign(new_bytes, 0);
    checkCudaErrors(cudaFree(d_image));
    checkCudaErrors(cudaMalloc((void**)&d_image, new_bytes));
    glDeleteTextures(1, &tex_id);
    create_texture();
    
    // cuda_scene.update_camera(image_width, image_height, samples_per_pixel, cuda_scene.h_cam.vfov, cuda_scene.h_cam.lookfrom, cuda_scene.h_cam.lookat, cuda_scene.h_cam.vup);
    
    // Trigger a new render after resizing
    // render_frame(kernel_choice);
}

void Renderer::update_camera(Point lookfrom, Point lookat, Vector camup, double vfov) {
    cuda_scene.update_camera(image_width, image_height, samples_per_pixel, vfov, lookfrom, lookat, camup);
}


////////////////////////////////////// SCENE //////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////

#endif // RENDERER_H