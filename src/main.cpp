/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-26
*/


#include "gobot/editor/main_window.hpp"
#include <QApplication>

int main(int argc, char **argv)
{
    QApplication app(argc, argv);
    gobot::main::MainWindow main_window;
    main_window.show();
    return QApplication::exec();
}
