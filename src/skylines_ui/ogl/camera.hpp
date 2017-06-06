#ifndef SKYLINES_OGL_CAMERA_HPP
#define SKYLINES_OGL_CAMERA_HPP
#include <QMatrix4x4>

namespace sl { namespace ui { namespace ogl {

    class Camera {
    public:
        enum class Type { PERSPECTIVE, ORTHOGRAPHIC };

        Camera();

        void Set();
        void Move(QVector3D delta);
        void Rotate(float angle, const QVector3D& axis);

        void LookAt(const QVector3D& eye, const QVector3D& center, const QVector3D& up);
        void SetOrthographic(float left, float right, float top, float bottom, float near_plane, float far_plane);
        void SetPerspective(float fov, float aspect, float near_plane, float far_plane);
    private:

        void UpdateProjectionMatrix();
        void UpdateViewMatrix();
        QVector3D GetLocalVector(const QVector3D &v);

        Type type_;
        //properties of the projection of the camera
        float fov_;          //view angle
        float aspect_;       //aspect ratio
        float near_plane_;   //near plane
        float far_plane_;    //far plane

        //for orthogonal projection
        float left_, right_, top_, bottom_;

        QVector3D eye_;
        QVector3D center_;
        QVector3D up_;

        QMatrix4x4 projection_;
        QMatrix4x4 view_;
    };
}}}
#endif // !SKYLINES_OGL_CAMERA_HPP
