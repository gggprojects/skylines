#ifndef SKYLINES_MAINWINDOW_HPP
#define SKYLINES_MAINWINDOW_HPP

#include <QMainWindow>
#include <QListWidgetItem>

#include "ogl/ogl_widget.hpp"
#include "queries/weighted.hpp"
#include "error/error_handler.hpp"

namespace Ui {
    class MainWindow;
}

namespace sl { namespace ui { namespace main_window {
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

        void RunSingleThreadBruteForce();
        void RunSingleThreadBruteForceWithDiscarting();
        void RunSingleThreadSorting();
        void RunMultiThreadBruteForce();
        void RunGPUBruteForce();

        void SerializeInputPoints();
        void LoadInputPoints();
        void Clear();
        void ClearResult();

        void MouseMoved(int dx, int dy);
        void PointSelected(int x, int y);

        void RemoveSelectedPoint(QListWidgetItem *item);

        Ui::MainWindow *ui_;
        ogl::OGLWidget *ogl_;

        sl::error::ThreadErrors_ptr thread_errors_ptr_;
        std::shared_ptr<sl::queries::WeightedQuery> weighted_query_ptr_;
    };
}}}
#endif // SKYLINES_MAINWINDOW_HPP
