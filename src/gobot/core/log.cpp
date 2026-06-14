/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 2021/4/7
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/log.hpp"

#include <cstdlib>
#include <system_error>

namespace gobot {
namespace {

std::filesystem::path GetLogFilePath() {
    const char* env_log_dir = std::getenv("GOBOT_LOG_DIR");
    if (env_log_dir != nullptr && *env_log_dir != '\0') {
        std::filesystem::path log_dir = env_log_dir;
        std::error_code error;
        std::filesystem::create_directories(log_dir, error);
        if (!error) {
            return log_dir / "gobot.log";
        }
    }

    const char* home = std::getenv("HOME");
    if (home == nullptr || *home == '\0') {
        return "gobot.log";
    }

    std::filesystem::path log_dir = std::filesystem::path(home) / ".gobot" / "log";
    std::error_code error;
    std::filesystem::create_directories(log_dir, error);
    if (error) {
        return "gobot.log";
    }

    return log_dir / "gobot.log";
}

} // namespace

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

    try {
        auto rotating_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            GetLogFilePath().string(), 1024 * 1024, 5, false);
        rotating_sink->set_pattern("[%T] [%@] [%l]: %v");
        sink_list.push_back(rotating_sink);
    } catch (const spdlog::spdlog_ex& error) {
        spdlog::warn("Gobot file logging disabled: {}", error.what());
    }
    logger_ = std::make_shared<spdlog::logger>("GOBOT", begin(sink_list), end(sink_list));

    //register it if you need to access it globally
    spdlog::register_logger(logger_);

    // set log level
    logger_->set_level(spdlog::level::trace);
    logger_->flush_on(spdlog::level::err);
}


}
