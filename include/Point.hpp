#ifndef POINT_H
#define POINT_H

#include "../common/common.hpp"
#include "../include/Vector.hpp"
#include <string>

class Point {
public:
    float m_x;
    float m_y;
    float m_z;

    __host__ __device__ Point(float x, float y, float z)
        : m_x(x), m_y(y), m_z(z) {}

    __host__ __device__ Point()
        : m_x(0), m_y(0), m_z(0) {}

    __host__ __device__ Point(const Point& other)
        : m_x(other.m_x), m_y(other.m_y), m_z(other.m_z) {}

    __host__ __device__ Point& operator=(const Point& other) {
        m_x = other.m_x;
        m_y = other.m_y;
        m_z = other.m_z;
        return *this;
    }

    __host__ __device__ Point operator+(const Vector& other) const {
        return Point(m_x + other.m_x, m_y + other.m_y, m_z + other.m_z);
    }

    __host__ __device__ Vector operator-(const Vector& other) const {
        return Vector(m_x - other.m_x, m_y - other.m_y, m_z - other.m_z);
    }

    __host__ __device__ Vector operator-(const Point& other) const {
        return Vector(m_x - other.m_x, m_y - other.m_y, m_z - other.m_z);
    }

    __host__ __device__ Vector to_vector() const {
        return Vector(m_x, m_y, m_z);
    }

    // Host-only: toString (not available on device)
    std::string toString() const {
        return "Point(x: " + std::to_string(m_x) + ", y: " + std::to_string(m_y) + ", z: " + std::to_string(m_z) + ")";
    }
};

#endif // POINT_H