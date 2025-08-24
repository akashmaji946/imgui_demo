#ifndef CUDA_SCENE_H
#define CUDA_SCENE_H

#include "../common/common.hpp"
#include "../common/OFFReader.h"
#include "../include/Camera.hpp"
#include "../include/Scene.hpp" 
#include "../include/Sphere.hpp"
#include "../include/Triangle.hpp"
#include "../include/Point.hpp"
#include "../include/Vector.hpp"
#include "../include/Color.hpp"
#include <iostream>


struct CudaScene {
    Camera h_cam;
    Camera* d_cam = nullptr;
    Object* d_objects = nullptr;
    Scene h_scene{};
    Scene* d_scene = nullptr;

    void init(int image_width, int image_height, int spp,  OFFModel* model) {

        h_cam = Camera(static_cast<double>(image_width)/image_height, image_width, image_height, spp, 20.0, Point(0, 0, 0), Point(0, 0, -1), Vector(0, 1, 0));
        h_cam.init();
        checkCudaErrors(cudaMalloc((void**)&d_cam, sizeof(Camera)));
        checkCudaErrors(cudaMemcpy(d_cam, &h_cam, sizeof(Camera), cudaMemcpyHostToDevice));
    

        // create triangles
        const int num_objects = model->numberOfPolygons;
        std::vector<Object> h_objects(num_objects);
        for(int i = 0; i < num_objects; i++) {
            h_objects[i].type = OBJ_TRIANGLE;
            h_objects[i].triangle = Triangle(
                Vector(model->vertices[model->polygons[i].v[0]].x, model->vertices[model->polygons[i].v[0]].y, model->vertices[model->polygons[i].v[0]].z),
                Vector(model->vertices[model->polygons[i].v[1]].x, model->vertices[model->polygons[i].v[1]].y, model->vertices[model->polygons[i].v[1]].z),
                Vector(model->vertices[model->polygons[i].v[2]].x, model->vertices[model->polygons[i].v[2]].y, model->vertices[model->polygons[i].v[2]].z)
            );
        }
        // FreeOffModel(model);

        checkCudaErrors(cudaMalloc((void**)&d_objects, sizeof(Object) * num_objects));
        checkCudaErrors(cudaMemcpy(d_objects, h_objects.data(), sizeof(Object) * num_objects, cudaMemcpyHostToDevice));

        h_scene.objects = d_objects;
        h_scene.num_objects = num_objects;
        checkCudaErrors(cudaMalloc((void**)&d_scene, sizeof(Scene)));
        checkCudaErrors(cudaMemcpy(d_scene, &h_scene, sizeof(Scene), cudaMemcpyHostToDevice));

    }

    void init(int image_width, int image_height, int spp) {
        std::cout << "Using default simple scene.\n";
        h_cam = Camera(static_cast<double>(image_width)/image_height, image_width, image_height, spp, 20.0, Point(0, 0, 0), Point(0, 0, -1), Vector(0, 1, 0));
        h_cam.init();
        checkCudaErrors(cudaMalloc((void**)&d_cam, sizeof(Camera)));
        checkCudaErrors(cudaMemcpy(d_cam, &h_cam, sizeof(Camera), cudaMemcpyHostToDevice));

        // Create a more complex scene with more objects
        const int num_objects = 10;
        std::vector<Object> h_objects(num_objects);

        // Object 0: Large sphere at the center
        h_objects[0].type = OBJ_SPHERE;
        h_objects[0].sphere = Sphere(Point(0, 0, -5), 1.5f);

        // Object 1: Small sphere in the foreground, right
        h_objects[1].type = OBJ_SPHERE;
        h_objects[1].sphere = Sphere(Point(2, -0.5, -3), 0.5f);

        // Object 2: Small sphere in the foreground, left
        h_objects[2].type = OBJ_SPHERE;
        h_objects[2].sphere = Sphere(Point(-2, 0.5, -3), 0.5f);

        // Object 3: A tall triangle in the background, left
        h_objects[3].type = OBJ_TRIANGLE;
        h_objects[3].triangle = Triangle(
            Vector(-3.0f, -2.0f, -8.0f),
            Vector(-1.0f, -2.0f, -8.0f),
            Vector(-2.0f,  2.0f, -8.0f)
        );

        // Object 4: A wide triangle in the foreground, right
        h_objects[4].type = OBJ_TRIANGLE;
        h_objects[4].triangle = Triangle(
            Vector(1.0f, -1.0f, -2.0f),
            Vector(3.0f, -1.0f, -2.0f),
            Vector(2.0f,  0.5f, -2.0f)
        );
        
        // Object 5: Small sphere high up and far back
        h_objects[5].type = OBJ_SPHERE;
        h_objects[5].sphere = Sphere(Point(-3, 2, -10), 0.8f);

        // Object 6: Small sphere low and far back
        h_objects[6].type = OBJ_SPHERE;
        h_objects[6].sphere = Sphere(Point(3, -2, -10), 0.8f);

        // Object 7: A triangle at an angle
        h_objects[7].type = OBJ_TRIANGLE;
        h_objects[7].triangle = Triangle(
            Vector(0.0f, -1.5f, -2.5f),
            Vector(1.5f, -1.5f, -2.5f),
            Vector(0.75f, 0.0f, -3.5f)
        );

        // Object 8: A small sphere close to the camera
        h_objects[8].type = OBJ_SPHERE;
        h_objects[8].sphere = Sphere(Point(-1.0, 1.0, -1.0), 0.3f);
        
        // Object 9: Another triangle, rotated
        h_objects[9].type = OBJ_TRIANGLE;
        h_objects[9].triangle = Triangle(
            Vector(-1.0f, -0.5f, -5.0f),
            Vector(1.0f, -0.5f, -5.0f),
            Vector(0.0f, 1.5f, -5.0f)
        );

        checkCudaErrors(cudaMalloc((void**)&d_objects, sizeof(Object) * num_objects));
        checkCudaErrors(cudaMemcpy(d_objects, h_objects.data(), sizeof(Object) * num_objects, cudaMemcpyHostToDevice));

        h_scene.objects = d_objects;
        h_scene.num_objects = num_objects;
        checkCudaErrors(cudaMalloc((void**)&d_scene, sizeof(Scene)));
        checkCudaErrors(cudaMemcpy(d_scene, &h_scene, sizeof(Scene), cudaMemcpyHostToDevice));
    }

    void update_camera(int image_width, int image_height, int spp, double vfov, Point lookfrom, Point lookat, Vector vup) {
        h_cam.set_camera(image_width, image_height, spp, lookfrom, lookat, vfov, vup);
        checkCudaErrors(cudaMemcpy(d_cam, &h_cam, sizeof(Camera), cudaMemcpyHostToDevice));
    }

    void destroy() {
        if (d_cam) cudaFree(d_cam), d_cam = nullptr;
        if (d_objects) cudaFree(d_objects), d_objects = nullptr;
        if (d_scene) cudaFree(d_scene), d_scene = nullptr;
    }
};

#endif // CUDA_SCENE_H
