#include "ogl/camera.hpp"

#pragma warning(push, 0)
#include <QOpenGLFunctions>

#include <GL/glu.h>
#pragma warning(pop)

namespace sl { namespace ui { namespace ogl {
    Camera::Camera() {
        view_.setToIdentity();
    }

    void Camera::Set() {
        glMatrixMode(GL_MODELVIEW);
        glLoadMatrixf(view_.data());

        glMatrixMode(GL_PROJECTION);
        glLoadMatrixf(projection_.data());
    }

    QVector3D Camera::Unproject(const QVector2D &screen_normalized_point) {
        QMatrix4x4 viewProjection = view_ * projection_;
        QMatrix4x4 viewProjectionInverse = viewProjection.inverted();
        QVector3D p(screen_normalized_point.x(), screen_normalized_point.y(), 0);
        return viewProjectionInverse * p;
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

    void Camera::LookAt(const QVector3D& eye, const QVector3D& center, const QVector3D& up) {
        eye_ = eye;
        center_ = center;
        up_ = up;

        UpdateViewMatrix();
    }

    QVector3D rotateVector(const QVector3D& v, const QMatrix4x4 &view) {
        QMatrix4x4 temp = view;
        temp.data()[12] = 0;
        temp.data()[13] = 0;
        temp.data()[14] = 0;
        return temp * v;
    }

    QVector3D Camera::GetLocalVector(const QVector3D &v) {
        QMatrix4x4 iV = view_;
        iV = iV.inverted();
        return std::move(rotateVector(v, iV));
    }

    void Camera::Move(QVector3D delta) {
        QVector3D localDelta = GetLocalVector(delta);
        eye_ -= localDelta;
        center_ -= localDelta;
        UpdateViewMatrix();
    }

    //----------------------------------------
    //------------ Ortographic ---------------
    //----------------------------------------
    void OrtographicCamera::SetOrthographic(float left, float right, float top, float bottom, float near_plane, float far_plane) {
        left_ = left;
        right_ = right;
        top_ = top;
        bottom_ = bottom;
        near_plane_ = near_plane;
        far_plane_ = far_plane;

        UpdateProjectionMatrix();
    }

    void OrtographicCamera::UpdateProjectionMatrix() {
        //We activate the matrix we want to work: projection
        glMatrixMode(GL_PROJECTION);

        //We set it as identity
        glLoadIdentity();

        glOrtho(left_, right_, bottom_, top_, near_plane_, far_plane_);

        //upload to hardware
        glGetFloatv(GL_PROJECTION_MATRIX, projection_.data());
    }

    void OrtographicCamera::Zoom(float delta) {
        left_ += delta;
        right_ -= delta;
        top_ -= delta;
        bottom_ += delta;
        UpdateProjectionMatrix();
    }

    QVector4D OrtographicCamera::GetOrtographic() {
        return std::move(QVector4D(left_, right_, top_, bottom_));
    }

    void OrtographicCamera::Resize(int width, int heigth) {
        //We activate the matrix we want to work: projection
        glMatrixMode(GL_PROJECTION);

        //We set it as identity
        glLoadIdentity();

        glViewport(0, 0, width, heigth);

        glOrtho(left_, right_, bottom_, top_, near_plane_, far_plane_);

        //upload to hardware
        glGetFloatv(GL_PROJECTION_MATRIX, projection_.data());
    }

    //----------------------------------------
    //------------ Perspective ---------------
    //----------------------------------------
    void PerspectiveCamera::SetPerspective(float fov, float aspect, float near_plane, float far_plane) {
        fov_ = fov;
        aspect_ = aspect;
        near_plane_ = near_plane;
        far_plane_ = far_plane;

        UpdateProjectionMatrix();
    }

    void PerspectiveCamera::UpdateProjectionMatrix() {
        //We activate the matrix we want to work: projection
        glMatrixMode(GL_PROJECTION);

        //We set it as identity
        glLoadIdentity();

        gluPerspective(fov_, aspect_, near_plane_, far_plane_);

        //upload to hardware
        glGetFloatv(GL_PROJECTION_MATRIX, projection_.data());
    }

    void PerspectiveCamera::Rotate(float angle, const QVector3D& axis) {
        QMatrix4x4 R;
        R.rotate(angle, axis);
        QVector3D new_front = R * (center_ - eye_);
        center_ = eye_ + new_front;
        UpdateViewMatrix();
    }

    void PerspectiveCamera::Resize(int width, int heigth) {
        //We activate the matrix we want to work: projection
        glMatrixMode(GL_PROJECTION);

        //We set it as identity
        glLoadIdentity();

        glViewport(0, 0, width, heigth);

        gluPerspective(fov_, aspect_, near_plane_, far_plane_);

        //upload to hardware
        glGetFloatv(GL_PROJECTION_MATRIX, projection_.data());
    }

    void PerspectiveCamera::Zoom(float delta) {
        Move(QVector3D(0, 0, -delta));
    }
}}}
