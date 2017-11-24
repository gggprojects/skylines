#ifndef SKYLINES_MAINWINDOW_HPP
#define SKYLINES_MAINWINDOW_HPP

#pragma warning(push, 0)
#include <QMainWindow>
#include <QListWidgetItem>
#pragma warning(pop)

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
    //signals:
        //void Run(queries::WeightedQuery::AlgorithmType algorithm_type);
        //void SetTopK(int top_k);

    public slots:
        void SetTopK(int top_k);

    private:
        void InitRandomP();
        void InitRandomQ();
        void InitRandom();
        void SaveImage();

        void RunSingleThreadBruteForce();
        void RunSingleThreadBruteForceWithDiscarting();
        void RunSingleThreadSorting();
        void RunMultiThreadBruteForceDiscarting();
        void RunGPUBruteForce();

        void SerializeInputPoints();
        void LoadInputPoints();
        void Clear();
        void ClearResult();

        void MouseMoved(int dx, int dy);
        void PointSelected(int x, int y);

        void RemoveSelectedPoint(QListWidgetItem *item);
        void DistanceTypeChanged(bool checked);

        Ui::MainWindow *ui_;
        ogl::OGLWidget *ogl_;
        queries::algorithms::DistanceType distance_type_;

        std::shared_ptr<sl::queries::WeightedQuery> weighted_query_ptr_;
    };
}}}
#endif // SKYLINES_MAINWINDOW_HPP
