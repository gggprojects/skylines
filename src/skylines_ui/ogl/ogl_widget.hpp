#ifndef SKYLINES_OGL_OGLWIDGET_HPP
#define SKYLINES_OGL_OGLWIDGET_HPP

#include <memory>

#include <QWidget>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>

#include "ogl/camera.hpp"
#include "common/irenderable.hpp"
#include "error/error_handler.hpp"

namespace sl { namespace ui { namespace ogl {

    enum class CursorMode {
        MOVE,
        SELECT
    };

    class OGLWidget :
        public QOpenGLWidget,
        protected QOpenGLFunctions,
        public error::ErrorHandler {
        Q_OBJECT

    public:
        OGLWidget(
            std::shared_ptr<sl::common::IRenderable> renderable_ptr, QWidget *parent = 0);
        ~OGLWidget();

        int heightForWidth(int w) const override { return w; }
        QImage GetFrameBufferImage();
        void SetCursorMode(CursorMode new_mode) {
            cursor_mode_ = new_mode;
        }
        void PaintBisector(const QVector2D &a, const QVector2D &b);
        QVector3D Unproject(const QVector2D &screen_point);

    public slots:
        void Cleanup();

    signals:
        void Moved(int dx, int dy);
        void Selected(int dx, int dy);
        void Painted();

    protected:
        void initializeGL() override;
        void paintGL() override;
        void resizeGL(int width, int height) override;
        void mousePressEvent(QMouseEvent *event) override;
        void mouseMoveEvent(QMouseEvent *event) override;
        void wheelEvent(QWheelEvent *event) override;

    private:
        QPoint lastPos_;

        OrtographicCamera camera_;

        std::shared_ptr<sl::common::IRenderable> renderable_ptr_;
        CursorMode cursor_mode_;
    };
}}}
#endif // SKYLINES_OGLWIDGET_HPP
