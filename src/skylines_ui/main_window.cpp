
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
        connect(ui_->clear_input_points_PB2, &QPushButton::clicked, this, &MainWindow::ClearInputPoints);
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

    std::string ToJson(const std::vector<queries::data::Point> &input_points) {
        rapidjson::StringBuffer sb;
        rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(sb);
        writer.StartObject();
        writer.String("points");
        writer.StartArray();
        for (const queries::data::Point &p : input_points) {
            writer.Double(p.x_); writer.Double(p.y_);
        }
        writer.EndArray();
        writer.EndObject();
        return std::move(std::string(sb.GetString()));
    }

    void ToFile(const std::string &filename, const std::string &json) {
        std::ofstream out(filename);
        out << json;
        out.close();
    }

    void MainWindow::SerializeInputPoints() {
        QString filename = QFileDialog::getSaveFileName(this, "Save File", qgetenv("HOME"), "json file(*.json)");
        if (!filename.endsWith(".json")) {
            filename.append(".json");
        }

        const std::vector<queries::data::Point> &input_points = weighted_query_ptr_->GetInputData().GetPoints();
        std::string json = ToJson(input_points);
        ToFile(filename.toStdString(), json);
    }

    std::string ReadAllFile(std::string filename) {
        std::ifstream t(filename);
        std::string json_str((std::istreambuf_iterator<char>(t)),
            std::istreambuf_iterator<char>());
        return std::move(json_str);
    }

    void ParseJson(const std::string &json_str, std::vector<queries::data::Point> *input_points) {
        rapidjson::Document document;
        document.Parse(json_str.c_str());

        if (document.HasParseError()) {
            return;
        }

        const rapidjson::Value& array = document["points"];
        assert(array.IsArray());

        rapidjson::Value::ConstValueIterator it = array.Begin();
        while (it != array.End()) {
            float x = static_cast<float>(it->GetDouble());
            ++it;
            float y = static_cast<float>(it->GetDouble());
            ++it;
            input_points->emplace_back(queries::data::Point(x, y));
        }
    }

    void MainWindow::LoadInputPoints() {
        QString filename = QFileDialog::getOpenFileName(this, "Open File", qgetenv("HOME"), "json file(*.json)");
        if (filename.isEmpty()) {
            return;
        }

        std::string json_str = ReadAllFile(filename.toStdString());
        std::vector<queries::data::Point> input_points;
        ParseJson(json_str, &input_points);
        weighted_query_ptr_->SetInputData(queries::InputData(std::move(input_points)));
        ogl_->update();
    }

    void MainWindow::ClearInputPoints() {
        weighted_query_ptr_->ClearInputData();
        ogl_->update();
    }
}}
