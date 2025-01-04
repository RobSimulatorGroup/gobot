/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 2021/4/7
*/

#include "gobot/log.hpp"

namespace gobot {

Logger::Logger() {
  Init();
}

Logger::~Logger() {
  spdlog::shutdown();
}

Logger& Logger::GetInstance() {
    static Logger instance;
    return instance;
}

void Logger::Init() {
    // initialize spdlog
    std::vector<spdlog::sink_ptr> sink_list;
    //add stdout sink
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_pattern("%^[%T] [%@]: %v%$");
    sink_list.push_back(console_sink);

    // add rotate file sink
    auto rotating_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>("gobot.log", 1024 * 1024,
                                                                           5, false);
    rotating_sink->set_pattern("[%T] [%@] [%l]: %v");
    sink_list.push_back(rotating_sink);
    logger_ = std::make_shared<spdlog::logger>("GOBOT", begin(sink_list), end(sink_list));

    //register it if you need to access it globally
    spdlog::register_logger(logger_);

    // set log level
    logger_->set_level(spdlog::level::trace);
    logger_->flush_on(spdlog::level::err);
}


}


