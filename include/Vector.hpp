#ifndef VECTOR_H
#define VECTOR_H

#include "../common/common.hpp"
#include <cmath>
#include <string>

class Point;

class Vector {
public:
    float m_x;
    float m_y;
    float m_z;

    __host__ __device__ Vector(float x, float y, float z)
        : m_x(x), m_y(y), m_z(z) {}

    __host__ __device__ Vector()
        : m_x(0), m_y(0), m_z(0) {}

    __host__ __device__ Vector(const Vector& other)
        : m_x(other.m_x), m_y(other.m_y), m_z(other.m_z) {}

    __host__ __device__ Vector& operator=(const Vector& other) {
        m_x = other.m_x;
        m_y = other.m_y;
        m_z = other.m_z;
        return *this;
    }

    __host__ __device__ Vector operator*(double t) const {
        return Vector(m_x * t, m_y * t, m_z * t);
    }

    __host__ __device__ Vector operator/(float t) const {
        if (t == 0) {
            return Vector(0, 0, 0); // Avoid division by zero
        }
        return Vector(m_x / t, m_y / t, m_z / t);
    }

    __host__ __device__ Vector operator-(Vector other) const {
        return Vector(m_x - other.m_x, m_y - other.m_y, m_z - other.m_z);
    }

    __host__ __device__ Vector operator+(Vector other) const {
        return Vector(m_x + other.m_x, m_y + other.m_y, m_z + other.m_z);
    }

    __host__ __device__ Vector operator-() const {
        return Vector(-m_x, -m_y, -m_z);
    }

    __host__ __device__ float length() const {
        return sqrtf(m_x * m_x + m_y * m_y + m_z * m_z);
    }

    __host__ __device__ Vector unit_vector() const {
        float len = length();
        return *this / len;
    }

    __host__ __device__ float dot(Vector other) const {
        return m_x * other.m_x + m_y * other.m_y + m_z * other.m_z;
    }

    __host__ __device__ float dot(Point other) const; // Define if Point is available

    __host__ __device__ Vector cross(const Vector& other) const {
        return Vector(
            m_y * other.m_z - m_z * other.m_y,
            m_z * other.m_x - m_x * other.m_z,
            m_x * other.m_y - m_y * other.m_x
        );
    }

    // Host-only: toString (not available on device)
    std::string toString() const {
        return "Vector(" + std::to_string(m_x) + ", " + std::to_string(m_y) + ", " + std::to_string(m_z) + ")";
    }

    Vector normalized() const {
        return this->unit_vector();
    }
};


inline Vector cross(const Vector& u, const Vector& v) {
    return u.cross(v);
}

inline float dot(const Vector& u, const Vector& v) {
    return u.dot(v);
}


// If you want to define dot(Point) here, you need the Point class definition.
// Otherwise, implement it in Point.hpp as an inline function.

#endif // VECTOR_H