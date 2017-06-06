
#include "main_window.hpp"
#include "ui_main_window.h"


namespace sl { namespace ui {
    MainWindow::MainWindow(QWidget *parent) :
        QMainWindow(parent),
        ui_(new Ui::MainWindow),
        thread_errors_ptr_(sl::error::ThreadsErrors::Instanciate()),
        weighted_query_(thread_errors_ptr_) {
        ui_->setupUi(this);
        ogl_ = new ogl::OGLWidget(this);
        ui_->central_widget_HLayout->addWidget(ogl_);
        connect(ui_->pushButton, &QPushButton::clicked, this, &MainWindow::Run);
    }

    MainWindow::~MainWindow() {
        delete ui_;
    }

    void MainWindow::Run() {
        weighted_query_.Run();
    }
}}