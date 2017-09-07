#pragma warning(push, 0)
#include <QMouseEvent>
#include <QCoreApplication>

#include <freeglut/GL/freeglut.h>
#pragma warning(pop)

#include "ogl/ogl_widget.hpp"

namespace sl { namespace ui { namespace ogl {
    OGLWidget::OGLWidget(
        std::shared_ptr<sl::common::IRenderable> renderable_ptr, QWidget *parent) :
        error::ErrorHandler("ogl", "info"),
        QOpenGLWidget(parent), renderable_ptr_(renderable_ptr), cursor_mode_(CursorMode::MOVE) {
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

        int argc = 0;
        char *argv = nullptr;
        glutInit(&argc, &argv);
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

        if (event->buttons() & Qt::RightButton) {
            emit Selected(event->x(), event->y());
        }
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

        emit Painted();
    }

    void OGLWidget::PaintBisector(const QVector2D &a, const QVector2D &b) {
        QVector2D m = (a + b) / 2;
        QVector2D ab_v = b - a;
        QVector2D ab_v_2(-ab_v.y(), ab_v.x());

        float x1 = -10;
        float y1 = ((x1 - m.x()) / ab_v_2.x()) * ab_v_2.y() + m.y();

        float x2 = 10;
        float y2 = ((x2 - m.x()) / ab_v_2.x()) * ab_v_2.y() + m.y();

        glBegin(GL_LINES);
        glVertex2d(x1, y1);
        glVertex2d(x2, y2);
        glEnd();
    }

    QVector3D OGLWidget::Unproject(const QVector2D &screen_point) {
        double x = 2.0 * screen_point.x() / 1024 - 1;
        double y = -2.0 * screen_point.y() / 1024 + 1;
        return std::move(camera_.Unproject(QVector2D(x, y)));
    }
}}}
