#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "../common/common.hpp"
#include "Ray.hpp"
#include "Vector.hpp"
#include "Point.hpp"
#include "SurfaceInteraction.hpp"
#include <string>

class Triangle
{
public:
    Vector v0;
    Vector v1;
    Vector v2;

    __host__ __device__ Triangle()
        : v0(), v1(), v2() {}

    __host__ __device__ Triangle(const Vector& _v0, const Vector& _v1, const Vector& _v2)
        : v0(_v0), v1(_v1), v2(_v2) {}

    // Return true if hit, else false
    __device__ bool intersect(Ray& r) const {
        Vector edge1 = v1 - v0;
        Vector edge2 = v2 - v0;
        Vector h = r.direction().cross(edge2);
        float a = edge1.dot(h);
        if (a > -1e-8 && a < 1e-8) {
            return false; // Ray is parallel to triangle
        }
        float f = 1.0f / a;
        Vector s = r.origin() - v0;
        float u = f * s.dot(h);
        if (u < 0.0f || u > 1.0f)
            return false; // Intersection is outside the triangle
        Vector q = s.cross(edge1);
        float v = f * r.direction().dot(q);
        if (v < 0.0f || u + v > 1.0f)
            return false; // Intersection is outside the triangle
        float t = f * edge2.dot(q);
        if (t > 1e-8) {
            return true; // Ray intersects triangle
        } else {
            return false; // Intersection is behind the ray origin
        }
    }

    __host__ __device__ bool intersect_si(const Ray& r, SurfaceInteraction& si) const {
        const float EPSILON = 1e-8f;
        Vector edge1 = v1 - v0;
        Vector edge2 = v2 - v0;
        Vector h = r.direction().cross(edge2);
        float a = edge1.dot(h);
        if (a > -EPSILON && a < EPSILON) return false; // Parallel
    
        float f = 1.0f / a;
        Vector s = r.origin() - v0;
        float u = f * s.dot(h);
        if (u < 0.0f || u > 1.0f) return false;
    
        Vector q = s.cross(edge1);
        float v = f * r.direction().dot(q);
        if (v < 0.0f || u + v > 1.0f) return false;
    
        float t = f * edge2.dot(q);
        if (t > EPSILON) {
            si.t = t;
            si.position = r.at(t).to_vector();
            si.hit = true;
            si.normal = edge1.cross(edge2).unit_vector();
            return true;
        }
        return false;
    }

    __host__ __device__ Vector normal() const {
        Vector edge1 = v1 - v0;
        Vector edge2 = v2 - v0;
        return edge1.cross(edge2).unit_vector();
    }

    __host__ __device__ Vector centroid() const {
        return (v0 + v1 + v2) / 3.0f;
    }

    // Host-only: toString (not available on device)
    std::string toString() const {
        return "Triangle(v0: " + v0.toString() + ", v1: " + v1.toString() + ", v2: " + v2.toString() + ")";
    }
};

#endif // TRIANGLE_H