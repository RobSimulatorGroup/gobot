
/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 24-3-24.
*/

#include "gobot/slam/eskf/eskf.hpp"
#include "gobot/slam/util/txt_io.hpp"

using namespace gobot::slam;
using namespace gobot;

int main(int argc, char *argv[]) {

    double antenna_angle =  12.06; // RTK天线安装偏角（角度)
    double antenna_pox_x = -0.17;  // RTK天线安装偏移X;
    double antenna_pox_y = -0.20;  // RTK天线安装偏移Y;

//    ESKFD eskf;
//
//    TxtIO io("/home/wqq/gobot/data/slam/imu_gnss_odom.txt");
//    Vector2d antenna_pos(antenna_pox_x, antenna_pox_y);
//
//    auto save_vec3 = [](std::ofstream& fout, const Vector3d& v) { fout << v[0] << " " << v[1] << " " << v[2] << " "; };
//    auto save_quat = [](std::ofstream& fout, const Quaterniond& q) {
//        fout << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " ";
//    };
//
//    auto save_result = [&save_vec3, &save_quat](std::ofstream& fout, const NavStated& save_state) {
//        fout << std::setprecision(18) << save_state.timestamp_ << " " << std::setprecision(9);
//        save_vec3(fout, save_state.p_);
//        save_quat(fout, save_state.R_.unit_quaternion());
//        save_vec3(fout, save_state.v_);
//        save_vec3(fout, save_state.bg_);
//        save_vec3(fout, save_state.ba_);
//        fout << std::endl;
//    };
//
//    std::ofstream fout("./data/ch3/gins.txt");
//    bool imu_inited = false, gnss_inited = false;
//
////    std::shared_ptr<sad::ui::PangolinWindow> ui = nullptr;
////    if (FLAGS_with_ui) {
////        ui = std::make_shared<sad::ui::PangolinWindow>();
////        ui->Init();
////    }
//
//    /// 设置各类回调函数
//    bool first_gnss_set = false;
//    Vector3d origin = Vector3d::Zero();
//
//    io.SetIMUProcessFunc([&](const IMU& imu) {
//                /// IMU 处理函数
//                if (!imu_init.InitSuccess()) {
//                    imu_init.AddIMU(imu);
//                    return;
//                }
//
//                /// 需要IMU初始化
//                if (!imu_inited) {
//                    // 读取初始零偏，设置ESKF
//                    ESKFD::Options options;
//                    // 噪声由初始化器估计
//                    options.gyro_var_ = sqrt(imu_init.GetCovGyro()[0]);
//                    options.acce_var_ = sqrt(imu_init.GetCovAcce()[0]);
//                    eskf.SetInitialConditions(options, imu_init.GetInitBg(), imu_init.GetInitBa(), imu_init.GetGravity());
//                    imu_inited = true;
//                    return;
//                }
//
//                if (!gnss_inited) {
//                    /// 等待有效的RTK数据
//                    return;
//                }
//
//                /// GNSS 也接收到之后，再开始进行预测
//                eskf.Predict(imu);
//
//                /// predict就会更新ESKF，所以此时就可以发送数据
//                auto state = eskf.GetNominalState();
//                if (ui) {
//                    ui->UpdateNavState(state);
//                }
//
//                /// 记录数据以供绘图
//                save_result(fout, state);
//
//                usleep(1e3);
//            })
//            .SetGNSSProcessFunc([&](const GNSS& gnss) {
//                /// GNSS 处理函数
//                if (!imu_inited) {
//                    return;
//                }
//
//                GNSS gnss_convert = gnss;
//                if (!sad::ConvertGps2UTM(gnss_convert, antenna_pos, FLAGS_antenna_angle) || !gnss_convert.heading_valid_) {
//                    return;
//                }
//
//                /// 去掉原点
//                if (!first_gnss_set) {
//                    origin = gnss_convert.utm_pose_.translation();
//                    first_gnss_set = true;
//                }
//                gnss_convert.utm_pose_.translation() -= origin;
//
//                // 要求RTK heading有效，才能合入ESKF
//                eskf.ObserveGps(gnss_convert);
//
//                auto state = eskf.GetNominalState();
//                if (ui) {
//                    ui->UpdateNavState(state);
//                }
//                save_result(fout, state);
//
//                gnss_inited = true;
//            })
//            .SetOdomProcessFunc([&](const Odom& odom) {
//                /// Odom 处理函数，本章Odom只给初始化使用
//                imu_init.AddOdom(odom);
//                if (FLAGS_with_odom && imu_inited && gnss_inited) {
//                    eskf.ObserveWheelSpeed(odom);
//                }
//            })
//            .Go();


    return 0;
}


