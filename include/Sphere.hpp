#ifndef SPHERE_H
#define SPHERE_H

#include "../common/common.hpp"
#include "Vector.hpp"
#include "Ray.hpp"
#include "Point.hpp"
#include "SurfaceInteraction.hpp"
#include <string>

class Sphere {
public:
    Point m_center;
    float m_radius;

    __host__ __device__ Sphere(const Point& center, float radius)
        : m_center(center), m_radius(radius) {}

    __host__ __device__ Sphere()
        : m_center(Point(0, 0, 0)), m_radius(1.0f) {}

    __host__ __device__ Sphere(const Sphere& other)
        : m_center(other.m_center), m_radius(other.m_radius) {}

    __host__ __device__ Sphere& operator=(const Sphere& other) {
        if (this != &other) {
            m_center = other.m_center;
            m_radius = other.m_radius;
        }
        return *this;
    }

    __host__ __device__ Point center() const {
        return m_center;
    }

    __host__ __device__ float radius() const {
        return m_radius;
    }

    __host__ __device__ Vector normal(const Point& point) {
        return (point - m_center.to_vector()).unit_vector();
    }

    // Host-only: toString (not available on device)
    std::string toString() const {
        return "Sphere(center: " + m_center.toString() + ", radius: " + std::to_string(m_radius) + ")";
    }

    __device__ bool intersect(Ray& r) {
        Vector oc = r.origin() - m_center;
        float a = r.direction().dot(r.direction());
        float b = 2.0f * oc.dot(r.direction());
        float c = oc.dot(oc) - m_radius * m_radius;
        float discriminant = b * b - 4 * a * c;
        return (discriminant > 0);
    }

    __host__ __device__ bool intersect_si(const Ray& r, SurfaceInteraction& si) const {
        Vector oc = r.origin() - m_center;
        float a = r.direction().dot(r.direction());
        float b = 2.0f * oc.dot(r.direction());
        float c = oc.dot(oc) - m_radius * m_radius;
        float discriminant = b * b - 4 * a * c;
        if (discriminant < 0) return false;
        float sqrt_disc = sqrtf(discriminant);
    
        // Find the nearest positive root
        float t0 = (-b - sqrt_disc) / (2.0f * a);
        float t1 = (-b + sqrt_disc) / (2.0f * a);
        float t = t0;
        if (t < 0) t = t1;
        if (t < 0) return false;
    
        si.t = t;
        si.position = r.at(t).to_vector();
        si.hit = true;
        si.normal = (si.position - m_center.to_vector()).unit_vector();
        return true;
    }
};

#endif // SPHERE_H