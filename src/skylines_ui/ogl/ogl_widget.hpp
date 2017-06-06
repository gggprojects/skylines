#ifndef SKYLINES_OGLWIDGET_HPP
#define SKYLINES_OGLWIDGET_HPP

#include <QWidget>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
//#include <QOpenGLBuffer>
#include <QMatrix4x4>

namespace sl { namespace ui { namespace ogl {
    class OGLWidget : public QOpenGLWidget, protected QOpenGLFunctions {
        Q_OBJECT
    public:
        OGLWidget(QWidget *parent = 0);
        ~OGLWidget();

    public slots:
        void SetXRotation(int angle);
        void SetYRotation(int angle);
        void SetZRotation(int angle);
        void Cleanup();

    signals:
        void XRotationChanged(int angle);
        void YRotationChanged(int angle);
        void ZRotationChanged(int angle);

    protected:
        void initializeGL() override;
        void paintGL() override;
        void resizeGL(int width, int height) override;
        void mousePressEvent(QMouseEvent *event) override;
        void mouseMoveEvent(QMouseEvent *event) override;

    private:
        int xRot_;
        int yRot_;
        int zRot_;

        QPoint lastPos_;

        QMatrix4x4 projection_;
        QMatrix4x4 camera_;
        QMatrix4x4 world_;
    };
}}}
#endif // SKYLINES_OGLWIDGET_HPP
