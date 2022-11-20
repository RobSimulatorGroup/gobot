
#include "gobot/main/main_window.hpp"
#include <QApplication>

int main(int argc, char **argv)
{
    QApplication app(argc, argv);
    gobot::main::MainWindow main_window;
    main_window.show();
    return QApplication::exec();
}
