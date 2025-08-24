#ifndef SCENE_H
#define SCENE_H

#include "Triangle.hpp"
#include "Sphere.hpp"

enum ObjectType { OBJ_TRIANGLE, OBJ_SPHERE };

struct Object {
    ObjectType type;
    union {
        Triangle triangle;
        Sphere sphere;
        // Add more shapes here
    };

    __host__ __device__
    Object() {} // <-- Add this line

    __host__ __device__
    bool intersect_si(const Ray& ray, SurfaceInteraction& si) const {
        switch (type) {
            case OBJ_TRIANGLE:
                return triangle.intersect_si(ray, si);
            case OBJ_SPHERE:
                return sphere.intersect_si(ray, si);
            default:
                return false;
        }
    }
};

struct Scene {
    Object* objects;
    int num_objects;
};

#endif // SCENE_H