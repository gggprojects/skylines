#include <QMouseEvent>
#include <QCoreApplication>

#include "ogl/ogl_widget.hpp"

namespace sl { namespace ui { namespace ogl {
    OGLWidget::OGLWidget(QWidget *parent)
        : QOpenGLWidget(parent) {
    }

    OGLWidget::~OGLWidget() {
        Cleanup();
    }

    void qNormalizeAngle(int &angle) {
        while (angle < 0)
            angle += 360 * 16;
        while (angle > 360 * 16)
            angle -= 360 * 16;
    }

    void OGLWidget::SetXRotation(int angle) {
        qNormalizeAngle(angle);
        if (angle != xRot_) {
            xRot_ = angle;
            emit XRotationChanged(angle);
            update();
        }
    }

    void OGLWidget::SetYRotation(int angle) {
        qNormalizeAngle(angle);
        if (angle != yRot_) {
            yRot_ = angle;
            emit YRotationChanged(angle);
            update();
        }
    }

    void OGLWidget::SetZRotation(int angle) {
        qNormalizeAngle(angle);
        if (angle != zRot_) {
            zRot_ = angle;
            emit ZRotationChanged(angle);
            update();
        }

    }
    
    void OGLWidget::initializeGL() {
        connect(context(), &QOpenGLContext::aboutToBeDestroyed, this, &OGLWidget::Cleanup);
        initializeOpenGLFunctions();

        QVector3D eye(0, 0, 0);
        QVector3D center(0, 0, -1);
        QVector3D up(0, 1, 0);
        camera_.LookAt(eye, center, up);

        camera_.SetOrthographic(-100, 100, 100, -100, -100, 100);
        glClearColor(1, 1, 1, 1);
        glEnable(GL_DEPTH_TEST);

        //glViewport(0, 0, 400, 300);
        //glMatrixMode(GL_PROJECTION);
        //glLoadIdentity();
        //glOrtho(0, 400, 0, 300, -1, 1);
        //glMatrixMode(GL_MODELVIEW);

        //glLoadIdentity();
        ////camera_.lookAt(QVector3D(0, 0, -2), QVector3D(0, 0, 0), QVector3D(0, 1, 0));
        ////projection_.perspective(70, 1, 0, 100);

        //glEnable(GL_DEPTH_TEST);
        //glEnable(GL_CULL_FACE);
    }

    void OGLWidget::Cleanup() {
        makeCurrent();
        //m_logoVbo.destroy();
        //delete m_program;
        //m_program = 0;
        doneCurrent();
    }

    void OGLWidget::resizeGL(int w, int h) {
        //glViewport(0, 0, w, h);
        //glMatrixMode(GL_PROJECTION);
        //glLoadIdentity();
        //glOrtho(0, w, 0, h, -1, 1);
        //glMatrixMode(GL_MODELVIEW);
        //glLoadIdentity();
        //glMatrixMode(GL_PROJECTION);
        //projection_.setToIdentity();
        //projection_.perspective(45.0f, GLfloat(w) / h, 0.0f, 100.0f);
        //glMatrixMode(GL_MODELVIEW);
    }

    void OGLWidget::mousePressEvent(QMouseEvent *event) {
        lastPos_ = event->pos();
    }

    void OGLWidget::mouseMoveEvent(QMouseEvent *event) {
        int dx = event->x() - lastPos_.x();
        int dy = event->y() - lastPos_.y();

        if (event->buttons() & Qt::LeftButton) {
            SetXRotation(xRot_ + 8 * dy);
            SetYRotation(yRot_ + 8 * dx);
        } else if (event->buttons() & Qt::RightButton) {
            SetXRotation(xRot_ + 8 * dy);
            SetZRotation(zRot_ + 8 * dx);
        }
        lastPos_ = event->pos();
    }

    void OGLWidget::wheelEvent(QWheelEvent *event) {
        //QPoint p = event->angleDelta();
        int a = event->delta();
        camera_
        //int numDegrees = 100 * direction;
        //float proportion = c->right - c->left;
        //float numSteps = ((numDegrees / 15.0)*0.008)*proportion;

    }

    void OGLWidget::paintGL() {
        camera_.Set();

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glColor3f(1, 0, 0);
        glBegin(GL_POLYGON); {
            glVertex3f(-50, -50, 1);
            glVertex2f(50, -50);
            glVertex2f(0, 50);
        }glEnd();

        //world_.setToIdentity();
        ////world_.rotate(180.0f - (xRot_ / 16.0f), 1, 0, 0);
        ////world_.rotate(yRot_ / 16.0f, 0, 1, 0);
        ////world_.rotate(zRot_ / 16.0f, 0, 0, 1);
        glColor3f(1, 0, 1);
        //unsigned int p[3];
        //p[0] = 0;
        //p[1] = 0;
        //p[2] = -1;
        //glDrawElements(GL_POINTS, 1, GL_UNSIGNED_INT, p);
        glBegin(GL_LINES); {
            glVertex2f(0, 0); glVertex2f(0.5, 0.5);
            glVertex2f(0, 0); glVertex2f(-0.5, -0.5);
            glVertex2f(0, 0); glVertex2f(0.5, -0.5);
            glVertex2f(0, 0); glVertex2f(-0.5, 0.5);
        }glEnd();
        //glEnd();
        //glDrawArrays(GL_TRIANGLES, 0, m_logo.vertexCount());
    }
}}}
