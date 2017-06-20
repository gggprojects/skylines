
#include <fstream>

#include <QFileDialog>
#include <QMessageBox>

#include <rapidjson/document.h>
#include <rapidjson/rapidjson.h>
#include <rapidjson/prettywriter.h>

#include "main_window.hpp"
#include "ui_main_window.h"

namespace sl { namespace ui {
    MainWindow::MainWindow(QWidget *parent) :
        QMainWindow(parent),
        ui_(new Ui::MainWindow),
        thread_errors_ptr_(sl::error::ThreadsErrors::Instanciate()) {
        weighted_query_ptr_ = std::make_shared<sl::queries::WeightedQuery>(thread_errors_ptr_);
        ui_->setupUi(this);
        ogl_ = new ogl::OGLWidget(weighted_query_ptr_, this);
        ui_->horizontalLayout_2->addWidget(ogl_);
        connect(ui_->pushButton, &QPushButton::clicked, this, &MainWindow::Run);
        connect(ui_->actionSave_image_to_file, &QAction::triggered, this, &MainWindow::SaveImage);
        connect(ui_->initRandom_PB, &QPushButton::clicked, this, &MainWindow::InitRandom);
        connect(ui_->serialize_input_points_PB, &QPushButton::clicked, this, &MainWindow::SerializeInputPoints);
        connect(ui_->load_input_points_PB, &QPushButton::clicked, this, &MainWindow::LoadInputPoints);
        connect(ui_->clear_PB2, &QPushButton::clicked, this, &MainWindow::Clear);
    }

    MainWindow::~MainWindow() {
        delete ui_;
    }

    void MainWindow::Run() {
        weighted_query_ptr_->Run();
    }

    void MainWindow::InitRandom() {
        weighted_query_ptr_->InitRandom(10);
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

    std::string ReadAllFile(std::string filename) {
        std::ifstream t(filename);
        std::string json_str((std::istreambuf_iterator<char>(t)),
            std::istreambuf_iterator<char>());
        return std::move(json_str);
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
}}
