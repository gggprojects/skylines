
#include <QFileDialog>
#include <QMessageBox>
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
        ui_->central_widget_HLayout->addWidget(ogl_);
        connect(ui_->pushButton, &QPushButton::clicked, this, &MainWindow::Run);
        connect(ui_->actionSave_image_to_file, &QAction::triggered, this, &MainWindow::SaveImage);
    }

    MainWindow::~MainWindow() {
        delete ui_;
    }

    void MainWindow::Run() {
        weighted_query_ptr_->Run();
    }

    void MainWindow::SaveImage() {
        QString filename = QFileDialog::getSaveFileName(this, "Save File", getenv("HOME"), "JPEG Image (*.jpg *.jpeg) ");
        if (!filename.endsWith(".jpg") && !filename.endsWith(".jpeg")) {
            filename.append(".jpg");
        }

        QImage image = ogl_->GetFrameBufferImage();
        if (!image.save(filename, "JPG")) {
            QMessageBox::warning(this, "Save Image", "Error saving image.");
        }
    }
}}