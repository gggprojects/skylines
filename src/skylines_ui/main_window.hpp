#ifndef SKYLINES_MAINWINDOW_HPP
#define SKYLINES_MAINWINDOW_HPP

#include <QMainWindow>

namespace Ui {
    class MainWindow;
}

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
};

#endif // SKYLINES_MAINWINDOW_HPP
