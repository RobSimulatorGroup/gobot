/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 2021/4/7
*/

#pragma once


#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <filesystem>

#include <spdlog/spdlog.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>


namespace gobot {

class Logger {
public:
    static Logger &getInstance() {
        static Logger instance;
        return instance;
    }

    auto getLogger() { return logger_; }
public:
    void init() {
        // initialize spdlog
        std::vector<spdlog::sink_ptr> sink_list;
        //add stdout sink
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        sink_list.push_back(console_sink);

        // add rotate file sink
        auto rotating_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>("gobot.log", 1024 * 1024,
                                                                               5, false);
        logger_ = std::make_shared<spdlog::logger>("GOBOT", begin(sink_list), end(sink_list));
        sink_list.push_back(rotating_sink);

        sink_list[0]->set_pattern("%^[%T] %n: %v%$");
        sink_list[1]->set_pattern("[%T] [%l] %n: %v");

        //register it if you need to access it globally
        spdlog::register_logger(logger_);

        // set log level
        logger_->set_level(spdlog::level::trace);
        logger_->flush_on(spdlog::level::trace);
    }

    Logger(const Logger &) = delete;
    Logger &operator=(const Logger &) = delete;

private:
    Logger() { init(); };
    ~Logger() { spdlog::shutdown(); };

    std::shared_ptr<spdlog::logger> logger_;
};

} // end of namespace gobot

#define SPDLOG_STR_H(x) #x
#define SPDLOG_STR_HELPER(x) SPDLOG_STR_H(x)

#define LOG_TRACE(...) gobot::Logger::getInstance().getLogger()->trace("[" __FILE__ ":" SPDLOG_STR_HELPER(__LINE__) "] ", __VA_ARGS__)
#define LOG_DEBUG(...) gobot::Logger::getInstance().getLogger()->debug("[" __FILE__ ":" SPDLOG_STR_HELPER(__LINE__) "] " __VA_ARGS__)
#define LOG_INFO(...)  gobot::Logger::getInstance().getLogger()->info("[" __FILE__ ": " SPDLOG_STR_HELPER(__LINE__) "] " __VA_ARGS__)
#define LOG_WARN(...)  gobot::Logger::getInstance().getLogger()->warn("[" __FILE__ ": " SPDLOG_STR_HELPER(__LINE__) "] " __VA_ARGS__)
#define LOG_ERROR(...) gobot::Logger::getInstance().getLogger()->error("[" __FILE__ ": " SPDLOG_STR_HELPER(__LINE__) "] " __VA_ARGS__)
#define LOG_FATAL(...) gobot::Logger::getInstance().getLogger()->critical("[" __FILE__ ": " SPDLOG_STR_HELPER(__LINE__) "] " __VA_ARGS__)