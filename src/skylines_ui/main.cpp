#include <QGuiApplication>
#include "ui_mainwindow.h"

int main(int argc, char *argv[]) {
    QGuiApplication app(argc, argv);

    Ui::MainWindow mw;
    //engine.load(QUrl(QStringLiteral("qrc:/main.qml")));
    //if (engine.rootObjects().isEmpty())
    //    return -1;
    
    return app.exec();
}
