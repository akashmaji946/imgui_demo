#ifndef RAY_H
#define RAY_H

#include "../common/common.hpp"
#include "Point.hpp"
#include "Vector.hpp"
#include <string>
#include <ostream>

class Ray {
public:
    Point m_origin;
    Vector m_direction;

    __host__ __device__ Ray()
        : m_origin(), m_direction() {}

    __host__ __device__ Ray(const Point& origin, const Vector& direction)
        : m_origin(origin), m_direction(direction) {}

    __host__ __device__ Ray(const Ray& other)
        : m_origin(other.m_origin), m_direction(other.m_direction) {}

    __host__ __device__ Point origin() const { return m_origin; }

    __host__ __device__ Vector direction() const { return m_direction; }

    __host__ __device__ Point at(double t) const {
        return m_origin + (m_direction * t);
    }

    __host__ __device__ Ray scaled(double t) const {
        return Ray(m_origin, m_direction * t);
    }

    __host__ __device__ Ray unit_ray() const {
        Vector unit_direction = m_direction.unit_vector();
        return Ray(m_origin, unit_direction);
    }

    // Host-only: toString (not available on device)
    std::string toString() const {
        return "Ray(origin: " + m_origin.toString() + ", direction: " + m_direction.toString() + ")";
    }

    // Optional: friend operator<< for host code
    friend std::ostream& operator<<(std::ostream& out, const Ray& ray) {
        out << ray.toString();
        return out;
    }
};

#endif // RAY_H