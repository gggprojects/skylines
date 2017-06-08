#ifndef SKYLINES_OGL_CAMERA_HPP
#define SKYLINES_OGL_CAMERA_HPP
#include <QMatrix4x4>

namespace sl { namespace ui { namespace ogl {

    class Camera {
    public:
        Camera();

        void Set();
        void LookAt(const QVector3D& eye, const QVector3D& center, const QVector3D& up);
        virtual void Resize(int width, int heigth) = 0;

        void Move(QVector3D delta);
        virtual void Zoom(float delta) = 0;
    protected:
        virtual void UpdateProjectionMatrix() = 0;
        void UpdateViewMatrix();
        QVector3D GetLocalVector(const QVector3D &v);

        float near_plane_;
        float far_plane_;

        QVector3D eye_;
        QVector3D center_;
        QVector3D up_;

        QMatrix4x4 projection_;
        QMatrix4x4 view_;
    };

    class OrtographicCamera : public Camera {
    public:
        OrtographicCamera() : Camera() {}

        void Zoom(float delta) final;

        void SetOrthographic(float left, float right, float top, float bottom, float near_plane, float far_plane);

        QVector4D GetOrtographic();

        void Resize(int width, int heigth) final;
    private:
        void UpdateProjectionMatrix() final;

        //for orthogonal projection
        float left_, right_, top_, bottom_;
    };

    class PerspectiveCamera : public Camera {
    public:
        PerspectiveCamera() : Camera() {}

        void Zoom(float delta) final;

        void Rotate(float angle, const QVector3D& axis);

        void SetPerspective(float fov, float aspect, float near_plane, float far_plane);

        void Resize(int width, int heigth) final;

    private:
        void UpdateProjectionMatrix() final;

        //properties of the projection of the camera
        float fov_;          //view angle
        float aspect_;       //aspect ratio
    };
}}}
#endif // !SKYLINES_OGL_CAMERA_HPP
