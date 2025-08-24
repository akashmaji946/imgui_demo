#ifndef COLOR_H
#define COLOR_H

#include "../common/common.hpp"
#include <string>

class Color {
public:
    float m_r;
    float m_g;
    float m_b;

    __host__ __device__ Color()
        : m_r(0), m_g(0), m_b(0) {}

    __host__ __device__ Color(float r, float g, float b)
        : m_r(r), m_g(g), m_b(b) {}

    __host__ __device__ Color(const Color& other)
        : m_r(other.m_r), m_g(other.m_g), m_b(other.m_b) {}

    __host__ __device__ Color& operator=(const Color& other) {
        m_r = other.m_r;
        m_g = other.m_g;
        m_b = other.m_b;
        return *this;
    }
    __host__ __device__ Color operator*(float t) const {
        return Color(m_r * t, m_g * t, m_b * t);
    }
    __host__ __device__ Color operator+(const Color& other) const {
        return Color(m_r + other.m_r, m_g + other.m_g, m_b + other.m_b);
    }
    __host__ __device__
    Color operator+=(const Color& other){
        m_r += other.m_r;
        m_g += other.m_g;
        m_b += other.m_b;
        return *this;
    }   
    __host__ __device__
    Color operator/=(double t){
        m_r /= t;
        m_g /= t;
        m_b /= t;
        return *this;
    }

    // Host-only: toString (not available on device)
    std::string toString() const {
        return "Color(r: " + std::to_string(m_r) + ", g: " + std::to_string(m_g) + ", b: " + std::to_string(m_b) + ")";
    }
};

#endif // COLOR_H