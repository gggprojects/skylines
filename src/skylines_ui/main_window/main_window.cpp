
#include <fstream>
#include <iostream>

#pragma warning(push, 0)
#include <QFileDialog>
#include <QMessageBox>
#include <QVector2D>

#include <rapidjson/document.h>
#include <rapidjson/rapidjson.h>
#include <rapidjson/prettywriter.h>
#pragma warning(pop)

#include "main_window.hpp"
#include "ui_main_window.h"
#include "queries/data/data_structures.hpp"

namespace sl { namespace ui { namespace main_window {
    MainWindow::MainWindow(QWidget *parent) :
        error::ErrorHandler("ui", "info"),
        QMainWindow(parent),
        ui_(new Ui::MainWindow),
        distance_type_(queries::algorithms::DistanceType::Neartest) {
        weighted_query_ptr_ = std::make_shared<sl::queries::WeightedQuery>();
        ui_->setupUi(this);
        ogl_ = new ogl::OGLWidget(weighted_query_ptr_, this);
        ui_->horizontalLayout_2->addWidget(ogl_);
        connect(ui_->pushButton_STS, &QPushButton::clicked, this, &MainWindow::RunSingleThreadSorting);
        connect(ui_->pushButton_STBF, &QPushButton::clicked, this, &MainWindow::RunSingleThreadBruteForce);
        connect(ui_->pushButton_STBFDiscarting, &QPushButton::clicked, this, &MainWindow::RunSingleThreadBruteForceWithDiscarting);
        connect(ui_->pushButton_MTBF, &QPushButton::clicked, this, &MainWindow::RunMultiThreadBruteForce);
        connect(ui_->pushButton_MTS, &QPushButton::clicked, this, &MainWindow::RunMultiThreadSorting);
        connect(ui_->pushButton_GPUBF, &QPushButton::clicked, this, &MainWindow::RunGPUBruteForce);

        connect(ui_->actionSave_image_to_file, &QAction::triggered, this, &MainWindow::SaveImage);
        connect(ui_->initRandom_PB, &QPushButton::clicked, this, &MainWindow::InitRandom);
        connect(ui_->serialize_input_points_PB, &QPushButton::clicked, this, &MainWindow::SerializeInputPoints);
        connect(ui_->load_input_points_PB, &QPushButton::clicked, this, &MainWindow::LoadInputPoints);
        connect(ui_->clear_PB2, &QPushButton::clicked, this, &MainWindow::Clear);
        connect(ui_->pushButton_ClearResult, &QPushButton::clicked, this, &MainWindow::ClearResult);
        connect(ogl_, &ogl::OGLWidget::Moved, this, &MainWindow::MouseMoved);
        connect(ogl_, &ogl::OGLWidget::Selected, this, &MainWindow::PointSelected);
        connect(ogl_, &ogl::OGLWidget::Painted, this, &MainWindow::Render);
        connect(ui_->listWidget_points, &QListWidget::itemSelectionChanged, this, &MainWindow::UpdateRender);
        connect(ui_->listWidget_points, &QListWidget::itemDoubleClicked, this, &MainWindow::RemoveSelectedPoint);
        connect(ui_->radioButton_Neartest, &QRadioButton::toggled, this, &MainWindow::DistanceTypeChanged);
    }

    MainWindow::~MainWindow() {
        delete ui_;
    }

    void MainWindow::RunSingleThreadBruteForce() {
        weighted_query_ptr_->RunAlgorithm(queries::WeightedQuery::AlgorithmType::SINGLE_THREAD_BRUTE_FORCE, distance_type_);
        ogl_->update();
    }

    void MainWindow::RunSingleThreadBruteForceWithDiscarting() {
        weighted_query_ptr_->RunAlgorithm(queries::WeightedQuery::AlgorithmType::SINGLE_THREAD_BRUTE_FORCE_DISCARTING, distance_type_);
        ogl_->update();
    }

    void MainWindow::RunSingleThreadSorting() {
        weighted_query_ptr_->RunAlgorithm(queries::WeightedQuery::AlgorithmType::SINGLE_THREAD_SORTING, distance_type_);
        ogl_->update();
    }

    void MainWindow::RunMultiThreadBruteForce() {
        weighted_query_ptr_->RunAlgorithm(queries::WeightedQuery::AlgorithmType::MULTI_THREAD_BRUTE_FORCE, distance_type_);
        ogl_->update();
    }

    void MainWindow::RunMultiThreadSorting() {
        //weighted_query_ptr_->FromFile("input_crash.json");
        weighted_query_ptr_->RunAlgorithm(queries::WeightedQuery::AlgorithmType::MULTI_THREAD_SORTING, distance_type_);
        ogl_->update();
    }

    void MainWindow::RunGPUBruteForce() {
        weighted_query_ptr_->RunAlgorithm(queries::WeightedQuery::AlgorithmType::GPU_BRUTE_FORCE, distance_type_);
        ogl_->update();
    }

    void MainWindow::InitRandom() {
        weighted_query_ptr_->Clear();
        weighted_query_ptr_->InitRandom(static_cast<size_t>(ui_->spinBox_P->value()), static_cast<size_t>(ui_->spinBox_Q->value()));
        ogl_->update();
    }

    void MainWindow::SaveImage() {
        QString filename = QFileDialog::getSaveFileName(this, "Save File", qgetenv("HOME"), "JPEG Image (*.jpg *.jpeg) ");
        if (!filename.endsWith(".jpg") && !filename.endsWith(".jpeg")) {
            filename.append(".jpg");
        }

        QImage image = ogl_->GetFrameBufferImage();
        if (!image.save(filename, "JPG")) {
            QMessageBox::warning(this, "Save Image", "Error saving image.");
        }
    }

    void MainWindow::SerializeInputPoints() {
        QString filename = QFileDialog::getSaveFileName(this, "Save File", qgetenv("HOME"), "Input File(*.json *.bin)");
        if (filename.isNull()) return;

        if (!filename.endsWith(".json") && !filename.endsWith(".bin")) {
            QMessageBox::warning(this, "Incorrect extension", "Incorrect extension");
            return;
        }

        if (!weighted_query_ptr_->ToFile(filename.toStdString())) {
            QMessageBox::warning(this, "Error when serializing points", "Error when serializing points");
        }
    }

    void MainWindow::LoadInputPoints() {
        QString filename = QFileDialog::getOpenFileName(this, "Open File", qgetenv("HOME"), "json file(*.json *.bin)");
        if (filename.isEmpty()) {
            return;
        }

        if (!filename.endsWith(".json") && !filename.endsWith(".bin")) {
            QMessageBox::warning(this, "Incorrect extension", "Incorrect extension");
            return;
        }

        if (!weighted_query_ptr_->FromFile(filename.toStdString())) {
            QMessageBox::warning(this, "Error at loading points", "Error at loading points");
        }

        ogl_->update();
    }

    void MainWindow::Clear() {
        weighted_query_ptr_->Clear();
        ogl_->update();
    }

    void MainWindow::ClearResult() {
        weighted_query_ptr_->ClearOutput();
        ogl_->update();
    }

    void MainWindow::MouseMoved(int dx, int dy) {
        //SL_LOG_INFO(std::to_string(dx) + "," + std::to_string(dy));
    }
    
    void MainWindow::PointSelected(int x, int y) {
        QVector3D projected_point = ogl_->Unproject(QVector2D(x, y));
        size_t input_point_position = weighted_query_ptr_->GetClosetsPointPosition(queries::data::Point(projected_point.x(), projected_point.y()));
        ui_->listWidget_points->addItem(QString::fromStdString(std::to_string(input_point_position)));
        ogl_->update();
    }

    void MainWindow::RemoveSelectedPoint(QListWidgetItem *item) {
        delete item;
    }

    void MainWindow::UpdateRender() {
        ogl_->update();
    }

    void MainWindow::Render() {
        //render added points
        for (int i = 0; i < ui_->listWidget_points->count(); ++i) {
            size_t position = ui_->listWidget_points->item(i)->text().toULong();
            const queries::data::WeightedPoint &wp = weighted_query_ptr_->GetInputP().GetPoints()[position];
            glColor3f(0, 1, 1);
            glPointSize(9);
            glBegin(GL_POINTS);
            glVertex2f(wp.point_.x_, wp.point_.y_);
            glEnd();
        }

        //render bisectors
        QList<QListWidgetItem*> items = ui_->listWidget_points->selectedItems();
        for (int i = 0; i < items.size() - 1; i++) {
            size_t a_pos = items[i]->text().toULong();
            size_t b_pos = items[i + 1]->text().toULong();
            const queries::data::WeightedPoint &wp_a = weighted_query_ptr_->GetInputP().GetPoints()[a_pos];
            const queries::data::WeightedPoint &wp_b = weighted_query_ptr_->GetInputP().GetPoints()[b_pos];
            ogl_->PaintBisector(QVector2D(wp_a.point_.x_, wp_a.point_.y_), QVector2D(wp_b.point_.x_, wp_b.point_.y_));
        }
    }

    void MainWindow::DistanceTypeChanged(bool checked) {
        distance_type_ = checked ? queries::algorithms::DistanceType::Neartest : queries::algorithms::DistanceType::Furthest;
    }
}}}
