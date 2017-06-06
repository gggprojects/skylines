#include "ogl/camera.hpp"
#include <QOpenGLFunctions>
#include <GL/glu.h>

namespace sl { namespace ui { namespace ogl {
    Camera::Camera() {
        type_ = Type::PERSPECTIVE;
        view_.setToIdentity();
    }

    void Camera::UpdateProjectionMatrix() {
        //We activate the matrix we want to work: projection
        glMatrixMode(GL_PROJECTION);

        //We set it as identity
        glLoadIdentity();

        if (type_ == Type::PERSPECTIVE)
            gluPerspective(fov_, aspect_, near_plane_, far_plane_);
        else
            glOrtho(left_, right_, bottom_, top_, near_plane_, far_plane_);

        //upload to hardware
        glGetFloatv(GL_PROJECTION_MATRIX, projection_.data());
    }

    void Camera::Set() {
        glMatrixMode(GL_MODELVIEW);
        glLoadMatrixf(view_.data());

        glMatrixMode(GL_PROJECTION);
        glLoadMatrixf(projection_.data());
    }

    void Camera::SetOrthographic(float left, float right, float top, float bottom, float near_plane, float far_plane) {
        type_ = Type::ORTHOGRAPHIC;

        left_ = left;
        right_ = right;
        top_ = top;
        bottom_ = bottom;
        near_plane_ = near_plane;
        far_plane_ = far_plane;

        UpdateProjectionMatrix();
    }

    void Camera::SetPerspective(float fov, float aspect, float near_plane, float far_plane) {
        type_ = Type::PERSPECTIVE;

        fov_ = fov;
        aspect_ = aspect;
        near_plane_ = near_plane;
        far_plane_ = far_plane;

        UpdateProjectionMatrix();
    }


    QVector3D Camera::GetLocalVector(const QVector3D &v) {
        QMatrix4x4 iV = view_;
        iV = iV.inverted();
        //QVector3D result = iV.rotate(v);
        //return std::move(result);
        return QVector3D();
    }

    void Camera::Move(QVector3D delta) {
        QVector3D localDelta = GetLocalVector(delta);
        eye_ = localDelta;
        center_ -= localDelta;
        UpdateViewMatrix();
    }

    void Camera::UpdateViewMatrix() {
        //We activate the matrix we want to work: modelview
        glMatrixMode(GL_MODELVIEW);

        //We set it as identity
        glLoadIdentity();

        //We find the look at matrix
        gluLookAt(eye_.x(), eye_.y(), eye_.z(), center_.x(), center_.y(), center_.z(), up_.x(), up_.y(), up_.z());

        //We get the matrix and store it in our app
        glGetFloatv(GL_MODELVIEW_MATRIX, view_.data());
    }

    void Camera::Rotate(float angle, const QVector3D& axis) {
        QMatrix4x4 R;
        R.rotate(angle, axis);
        QVector3D new_front = R * (center_ - eye_);
        center_ = eye_ + new_front;
        UpdateViewMatrix();
    }

    void Camera::LookAt(const QVector3D& eye, const QVector3D& center, const QVector3D& up) {
        eye_ = eye;
        center_ = center;
        up_ = up;

        UpdateViewMatrix();
    }
}}}
