#ifndef SKYLINES_MAINWINDOW_HPP
#define SKYLINES_MAINWINDOW_HPP

#include <QMainWindow>
#include "ogl/ogl_widget.hpp"
#include "queries/weighted.hpp"
#include "error/error_handler.hpp"

namespace Ui {
    class MainWindow;
}

namespace sl { namespace ui {
    class MainWindow : public QMainWindow, public error::ErrorHandler {
        Q_OBJECT

    public:
        explicit MainWindow(QWidget *parent = 0);
        ~MainWindow();

        void UpdateRender();
        void Render();
    private:
        void InitRandom();
        void SaveImage();
        void Run();
        void SerializeInputPoints();
        void LoadInputPoints();
        void Clear();

        void MoveToolToggled();
        void MouseMoved(int dx, int dy);
        void PointSelected(int x, int y);

        Ui::MainWindow *ui_;
        ogl::OGLWidget *ogl_;

        sl::error::ThreadErrors_ptr thread_errors_ptr_;
        std::shared_ptr<sl::queries::WeightedQuery> weighted_query_ptr_;
    };
}}
#endif // SKYLINES_MAINWINDOW_HPP
