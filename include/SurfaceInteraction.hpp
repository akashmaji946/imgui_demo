#ifndef SURFACE_INTERACTION_H
#define SURFACE_INTERACTION_H

#include "../common/common.hpp"
#include "Vector.hpp"


class SurfaceInteraction
{
public:
    __host__ __device__ SurfaceInteraction(){
        hit = false;
        normal = Vector(0, 0, 0);
        position = Vector(0, 0, 0);
        bsdf = Vector(0, 0, 0);
    }
    bool hit;
    Vector normal;
    Vector position;
    Vector bsdf;
    float t;
};

#endif // SURFACE_INTERACTION_H