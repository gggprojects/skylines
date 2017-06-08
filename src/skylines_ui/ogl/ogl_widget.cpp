#include <QMouseEvent>
#include <QCoreApplication>

#include "ogl/ogl_widget.hpp"

namespace sl { namespace ui { namespace ogl {
    OGLWidget::OGLWidget(std::shared_ptr<sl::common::IRenderable> renderable_ptr, QWidget *parent)
        : QOpenGLWidget(parent), renderable_ptr_(renderable_ptr) {
    }

    OGLWidget::~OGLWidget() {
        Cleanup();
    }

    void OGLWidget::initializeGL() {
        connect(context(), &QOpenGLContext::aboutToBeDestroyed, this, &OGLWidget::Cleanup);
        initializeOpenGLFunctions();

        QVector3D eye(0, 0, 0);
        QVector3D center(0, 0, -1);
        QVector3D up(0, 1, 0);
        camera_.LookAt(eye, center, up);
        camera_.SetOrthographic(0, 1, 1, 0, -10, 10);

        glClearColor(1, 1, 1, 1);
        glEnable(GL_DEPTH_TEST);
    }

    void OGLWidget::Cleanup() {
        makeCurrent();
        doneCurrent();
    }

    QImage OGLWidget::GetFrameBufferImage() {
        return std::move(grabFramebuffer());
    }

    void OGLWidget::resizeGL(int w, int h) {
        camera_.Resize(w, h);
        update();
    }

    void OGLWidget::mousePressEvent(QMouseEvent *event) {
        lastPos_ = event->pos();
    }

    void OGLWidget::mouseMoveEvent(QMouseEvent *event) {
        int dx = event->x() - lastPos_.x();
        int dy = event->y() - lastPos_.y();

        if (event->buttons() & Qt::LeftButton) {
            QVector4D ortho = camera_.GetOrtographic();
            float right = ortho.y();
            float proportion = ortho.y() - ortho.x();
            float speed = 0.001f * proportion;

            QVector3D movement(dx*right*speed, -dy*right*speed, 0);
            camera_.Move(movement);
            update();
        } else if (event->buttons() & Qt::RightButton) {
            //SetXRotation(xRot_ + 8 * dy);
            //SetZRotation(zRot_ + 8 * dx);
        }
        lastPos_ = event->pos();
    }

    void OGLWidget::wheelEvent(QWheelEvent *event) {
        int numDegrees = event->delta();
        QVector4D ortho = camera_.GetOrtographic();
        float proportion = ortho.y() - ortho.x();
        float numSteps = ((numDegrees / 15.0)*0.008)*proportion;
        camera_.Zoom(numSteps);
        update();
    }

    void OGLWidget::paintGL() {
        camera_.Set();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor3f(0, 0, 0);
        glBegin(GL_LINES); glVertex2f(0, 0); glVertex2f(0, 1); glEnd();
        glBegin(GL_LINES); glVertex2f(0, 0); glVertex2f(1, 0); glEnd();
        renderable_ptr_->Render();
    }
}}}
