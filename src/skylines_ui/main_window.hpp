#ifndef SKYLINES_MAINWINDOW_HPP
#define SKYLINES_MAINWINDOW_HPP

#include <QMainWindow>
#include "ogl/ogl_widget.hpp"
#include "queries/weighted.hpp"

namespace Ui {
    class MainWindow;
}

namespace sl { namespace ui {
    class MainWindow : public QMainWindow {
        Q_OBJECT

    public:
        explicit MainWindow(QWidget *parent = 0);
        ~MainWindow();

    private:
        void InitRandom();
        void SaveImage();
        void Run();
        void SerializeInputPoints();
        void LoadInputPoints();
        void Clear();

        Ui::MainWindow *ui_;
        ogl::OGLWidget *ogl_;

        sl::error::ThreadErrors_ptr thread_errors_ptr_;
        std::shared_ptr<sl::queries::WeightedQuery> weighted_query_ptr_;
    };
}}
#endif // SKYLINES_MAINWINDOW_HPP
