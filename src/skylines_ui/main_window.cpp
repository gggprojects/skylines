
#include <fstream>
#include <iostream>

#include <QFileDialog>
#include <QMessageBox>
#include <QVector2D>

#include <rapidjson/document.h>
#include <rapidjson/rapidjson.h>
#include <rapidjson/prettywriter.h>

#include "main_window.hpp"
#include "ui_main_window.h"
#include "queries/data/data_structures.hpp"

namespace sl { namespace ui {
    MainWindow::MainWindow(QWidget *parent) :
        error::ErrorHandler("ui", "info"),
        QMainWindow(parent),
        ui_(new Ui::MainWindow),
        thread_errors_ptr_(sl::error::ThreadsErrors::Instanciate()) {
        SetThreadErrors(thread_errors_ptr_);
        weighted_query_ptr_ = std::make_shared<sl::queries::WeightedQuery>(thread_errors_ptr_);
        ui_->setupUi(this);
        ogl_ = new ogl::OGLWidget(thread_errors_ptr_, weighted_query_ptr_, this);
        ui_->horizontalLayout_2->addWidget(ogl_);
        connect(ui_->pushButton, &QPushButton::clicked, this, &MainWindow::Run);
        connect(ui_->actionSave_image_to_file, &QAction::triggered, this, &MainWindow::SaveImage);
        connect(ui_->initRandom_PB, &QPushButton::clicked, this, &MainWindow::InitRandom);
        connect(ui_->serialize_input_points_PB, &QPushButton::clicked, this, &MainWindow::SerializeInputPoints);
        connect(ui_->load_input_points_PB, &QPushButton::clicked, this, &MainWindow::LoadInputPoints);
        connect(ui_->clear_PB2, &QPushButton::clicked, this, &MainWindow::Clear);
        connect(ui_->radioButton_Move, &QRadioButton::toggled, this, &MainWindow::MoveToolToggled);
        connect(ogl_, &ogl::OGLWidget::Moved, this, &MainWindow::MouseMoved);
        connect(ogl_, &ogl::OGLWidget::Selected, this, &MainWindow::PointSelected);
        connect(ogl_, &ogl::OGLWidget::Painted, this, &MainWindow::Render);
        connect(ui_->listWidget_points, &QListWidget::itemSelectionChanged, this, &MainWindow::UpdateRender);
        //connect(ui_->listWidget_points, &QListWidget::currentItemChanged, this, &MainWindow::UpdateRender);
    }

    MainWindow::~MainWindow() {
        delete ui_;
    }

    std::string ReadAllFile(const std::string &filename) {
        std::ifstream t(filename);
        std::string json_str((std::istreambuf_iterator<char>(t)),
            std::istreambuf_iterator<char>());
        return std::move(json_str);
    }

    void MainWindow::Run() {
        ////debug
        //std::string json_str = ReadAllFile("first_dominated.json");
        //weighted_query_ptr_->FromJson(json_str);

        weighted_query_ptr_->Run();
        ogl_->update();
    }

    void MainWindow::InitRandom() {
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

    void ToFile(const std::string &filename, const std::string &json) {
        std::ofstream out(filename);
        out << json;
        out.close();
    }

    void MainWindow::SerializeInputPoints() {
        QString filename = QFileDialog::getSaveFileName(this, "Save File", qgetenv("HOME"), "json file(*.json)");
        if (filename.isNull()) return;

        if (!filename.endsWith(".json")) {
            filename.append(".json");
        }

        std::string json = weighted_query_ptr_->ToJson();
        ToFile(filename.toStdString(), json);
    }

    void MainWindow::LoadInputPoints() {
        QString filename = QFileDialog::getOpenFileName(this, "Open File", qgetenv("HOME"), "json file(*.json)");
        if (filename.isEmpty()) {
            return;
        }

        std::string json_str = ReadAllFile(filename.toStdString());
        weighted_query_ptr_->FromJson(json_str);
        ogl_->update();
    }

    void MainWindow::Clear() {
        weighted_query_ptr_->Clear();
        ogl_->update();
    }

    void MainWindow::MoveToolToggled() {
        ogl_->SetCursorMode(ui_->radioButton_Move->isChecked() ? ogl::CursorMode::MOVE : ogl::CursorMode::SELECT);
    }

    void MainWindow::MouseMoved(int dx, int dy) {
        //SL_LOG_INFO(std::to_string(dx) + "," + std::to_string(dy));
    }
    
    void MainWindow::PointSelected(int x, int y) {
        QVector3D projected_point = ogl_->Unproject(QVector2D(x, y));
        size_t input_point_position = weighted_query_ptr_->GetClosetsPointPosition(queries::data::Point(projected_point.x(), projected_point.y()));
        ui_->listWidget_points->addItem(QString::fromStdString(std::to_string(input_point_position)));
    }

    void MainWindow::UpdateRender() {
        ogl_->update();
    }

    void MainWindow::Render() {
        //render selected points
        QList<QListWidgetItem*> items = ui_->listWidget_points->selectedItems();
        for (const QListWidgetItem* item : items) {
            size_t position = item->text().toULong();
            const queries::data::WeightedPoint &wp = weighted_query_ptr_->GetPoint(position);
            glColor3f(0, 1, 1);
            glPointSize(9);
            glBegin(GL_POINTS);
            glVertex2f(wp.point_.x_, wp.point_.y_);
            glEnd();
        }
        for (int i = 0; i < items.size() - 1; i++) {
            size_t a_pos = items[i]->text().toULong();
            size_t b_pos = items[i + 1]->text().toULong();
            const queries::data::WeightedPoint &wp_a = weighted_query_ptr_->GetPoint(a_pos);
            const queries::data::WeightedPoint &wp_b = weighted_query_ptr_->GetPoint(b_pos);
            ogl_->PaintBisector(QVector2D(wp_a.point_.x_, wp_a.point_.y_), QVector2D(wp_b.point_.x_, wp_b.point_.y_));
        }
    }
}}
