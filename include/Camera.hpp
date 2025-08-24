#ifndef CAMERA_H
#define CAMERA_H

#include "../common/common.hpp"
#include "../include/Vector.hpp"
#include "../include/Point.hpp"
#include "../include/Ray.hpp"

class Camera {
public:
    // Constructors
    __host__ __device__
    Camera()
        : aspect_ratio(16.0 / 9.0), image_width(800), image_height(450), samples_per_pixel(10),
          pixel_scale(1.0 / 10), INIT(false),
          lookfrom(Point(0, 0, 0)), lookat(Point(0, 0, -1)), vup(Vector(0, 1, 0)), vfov(20.0),
          camera_origin(Point(0, 0, 0)),
          viewport_u(Vector(0, 0, 0)), viewport_v(Vector(0, 0, 0)),
          d_u(Vector(0, 0, 0)), d_v(Vector(0, 0, 0)),
          upper_left_corner(Point(0, 0, 0)), pixel_origin(Point(0, 0, 0)) {}

    __host__ __device__
    Camera(double aspect_ratio, int image_width, int image_height, int samples_per_pixel, double vfov)
        : aspect_ratio(aspect_ratio), image_width(image_width), image_height(image_height), samples_per_pixel(samples_per_pixel),
          pixel_scale(1.0 / samples_per_pixel), INIT(false),
          lookfrom(Point(0, 0, 0)), lookat(Point(0, 0, -1)), vup(Vector(0, 1, 0)), vfov(vfov),
          camera_origin(Point(0, 0, 0)),
          viewport_u(Vector(0, 0, 0)), viewport_v(Vector(0, 0, 0)),
          d_u(Vector(0, 0, 0)), d_v(Vector(0, 0, 0)),
          upper_left_corner(Point(0, 0, 0)), pixel_origin(Point(0, 0, 0)) {}

    // New constructor with full camera controls
    __host__ __device__
    Camera(double aspect_ratio, int image_width, int image_height, int samples_per_pixel, double vfov, Point lookfrom, Point lookat, Vector vup)
        : aspect_ratio(aspect_ratio), image_width(image_width), image_height(image_height), samples_per_pixel(samples_per_pixel),
          pixel_scale(1.0 / samples_per_pixel), INIT(false),
          lookfrom(lookfrom), lookat(lookat), vup(vup), vfov(vfov),
          camera_origin(Point(0, 0, 0)),
          viewport_u(Vector(0, 0, 0)), viewport_v(Vector(0, 0, 0)),
          d_u(Vector(0, 0, 0)), d_v(Vector(0, 0, 0)),
          upper_left_corner(Point(0, 0, 0)), pixel_origin(Point(0, 0, 0)) {}

    // setup camera parameters
    __host__ __device__ void set_camera(int width, int height, int samples_per_pixel, Point lookfrom, Point lookat, double vfov, Vector vup) {
        this->aspect_ratio = static_cast<double>(width) / height;
        this->image_width = width;
        this->image_height = height;

        this->samples_per_pixel = samples_per_pixel;
        this->pixel_scale = 1.0 / samples_per_pixel;

        this->camera_origin = lookfrom;
        this->lookfrom = lookfrom;
        this->lookat = lookat;
        this->vfov = vfov;
        this->vup = vup;

        this->init();
    }


    // Methods
    __host__ __device__ void init() {
        // Calculate image height
        image_height = static_cast<int>(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        // Determine camera basis vectors
        Vector w = (lookfrom - lookat).unit_vector();
        Vector u = (vup.cross(w)).unit_vector();
        Vector v = w.cross(u);

        // Calculate viewport dimensions
        double theta = vfov * M_PI / 180.0;
        double h = tan(theta / 2.0);
        double viewport_height = 2.0 * h;
        double viewport_width = viewport_height * aspect_ratio;

        // Calculate the vectors for horizontal and vertical pixel deltas
        camera_origin = lookfrom;
        viewport_u =   u * viewport_width;
        viewport_v = (-v) * viewport_height;

        d_u = viewport_u / static_cast<double>(image_width);
        d_v = viewport_v / static_cast<double>(image_height);

        // Calculate the location of the upper-left pixel
        Vector upper_left = camera_origin - (w * 1.0) - viewport_u / 2.0 - viewport_v / 2.0;
        upper_left_corner = Point(upper_left.m_x, upper_left.m_y, upper_left.m_z);
        pixel_origin = upper_left_corner + d_u * 0.5 + d_v * 0.5;

        
    }

    __host__ __device__ Ray get_ray(int i, int j) const {
        Point pixel_center = pixel_origin + d_u * i + d_v * j;
        Vector ray_direction = pixel_center - camera_origin;
        return Ray(camera_origin, ray_direction.unit_vector());
    }

    __host__ __device__ void update_camera(int width, int height, int samples_per_pixel, double vfov, Point lookfrom, Point lookat) {
        this->aspect_ratio = static_cast<double>(width) / height;
        this->image_width = width;
        this->image_height = height;
        this->samples_per_pixel = samples_per_pixel;
        this->pixel_scale = 1.0 / samples_per_pixel;
        this->lookfrom = lookfrom;
        this->lookat = lookat;
        this->vfov = vfov;
        init();
    }

public:
    double aspect_ratio;
    int image_width;
    int image_height;
    int samples_per_pixel;
    double pixel_scale;
    bool INIT;

    Point lookfrom;
    Point lookat;
    Vector vup;
    double vfov;

    Point camera_origin;
    Vector viewport_u;
    Vector viewport_v;
    Vector d_u;
    Vector d_v;
    Point upper_left_corner;
    Point pixel_origin;
};

#endif // CAMERA_H