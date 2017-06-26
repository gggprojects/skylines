#ifndef SKYLINES_OGL_OGLWIDGET_HPP
#define SKYLINES_OGL_OGLWIDGET_HPP

#include <memory>

#include <QWidget>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>

#include "ogl/camera.hpp"
#include "common/irenderable.hpp"

namespace sl { namespace ui { namespace ogl {

    enum class CursorMode {
        MOVE,
        SELECT
    };

    class OGLWidget : public QOpenGLWidget, protected QOpenGLFunctions {
        Q_OBJECT

    public:
        OGLWidget(std::shared_ptr<sl::common::IRenderable> renderable_ptr, QWidget *parent = 0);
        ~OGLWidget();

        int heightForWidth(int w) const override { return w; }
        QImage GetFrameBufferImage();
    public slots:
        void Cleanup();

    protected:
        void initializeGL() override;
        void paintGL() override;
        void resizeGL(int width, int height) override;
        void mousePressEvent(QMouseEvent *event) override;
        void mouseMoveEvent(QMouseEvent *event) override;
        void wheelEvent(QWheelEvent *event) override;
        void Move(QMouseEvent *event, int dx, int dy);
    private:
        QPoint lastPos_;

        OrtographicCamera camera_;

        std::shared_ptr<sl::common::IRenderable> renderable_ptr_;
        CursorMode cursor_mode_;
    };
}}}
#endif // SKYLINES_OGLWIDGET_HPP
