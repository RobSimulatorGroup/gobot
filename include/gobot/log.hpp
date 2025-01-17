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
#include <spdlog/fmt/ostr.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include "gobot/core/types.hpp"


namespace gobot {

class Logger {
public:
    static Logger& GetInstance();

    auto GetLogger() { return logger_; }
public:
    void Init();

    Logger(const Logger &) = delete;
    Logger &operator=(const Logger &) = delete;

    void AddSink(spdlog::sink_ptr& sink)
    {
        sink->set_pattern("[%T] [%@]: %v");
        logger_->sinks().push_back(sink);
    }

private:
    Logger();
    ~Logger();

    std::shared_ptr<spdlog::logger> logger_;
};

} // end of namespace gobot

#define GOB_LOG_LEVEL_TRACE 0
#define GOB_LOG_LEVEL_DEBUG 1
#define GOB_LOG_LEVEL_INFO 2
#define GOB_LOG_LEVEL_WARN 3
#define GOB_LOG_LEVEL_ERROR 4
#define GOB_LOG_LEVEL_CRITICAL 5
#define GOB_LOG_LEVEL_OFF 6

#if !defined(GOB_LOG_ACTIVE_LEVEL)
#    define GOB_LOG_ACTIVE_LEVEL GOB_LOG_LEVEL_DEBUG
#endif


#if GOB_LOG_ACTIVE_LEVEL <= GOB_LOG_LEVEL_TRACE
#    define SPDLOG_LOGGER_TRACE(logger, ...) SPDLOG_LOGGER_CALL(logger, spdlog::level::trace, __VA_ARGS__)
#    define LOG_TRACE(...) SPDLOG_LOGGER_TRACE(gobot::Logger::GetInstance().GetLogger(), __VA_ARGS__)
#else
#    define SPDLOG_LOGGER_TRACE(logger, ...) (void)0
#    define LOG_TRACE(...) (void)0
#endif

#if GOB_LOG_ACTIVE_LEVEL <= GOB_LOG_LEVEL_DEBUG
#    define GOB_LOGGER_DEBUG(logger, ...) SPDLOG_LOGGER_CALL(logger, spdlog::level::debug, __VA_ARGS__)
#    define LOG_DEBUG(...) GOB_LOGGER_DEBUG(gobot::Logger::GetInstance().GetLogger(), __VA_ARGS__)
#else
#    define SPDLOG_LOGGER_DEBUG(logger, ...) (void)0
#    define LOG_DEBUG(...) (void)0
#endif

#if GOB_LOG_ACTIVE_LEVEL <= GOB_LOG_LEVEL_INFO
#    define GOB_LOGGER_INFO(logger, ...) SPDLOG_LOGGER_CALL(logger, spdlog::level::info, __VA_ARGS__)
#    define LOG_INFO(...) GOB_LOGGER_INFO(gobot::Logger::GetInstance().GetLogger(), __VA_ARGS__)
#else
#    define SPDLOG_LOGGER_INFO(logger, ...) (void)0
#    define LOG_INFO(...) (void)0
#endif

#if GOB_LOG_ACTIVE_LEVEL <= GOB_LOG_LEVEL_WARN
#    define GOB_LOGGER_WARN(logger, ...) SPDLOG_LOGGER_CALL(logger, spdlog::level::warn, __VA_ARGS__)
#    define LOG_WARN(...) GOB_LOGGER_WARN(gobot::Logger::GetInstance().GetLogger(), __VA_ARGS__)
#else
#    define SPDLOG_LOGGER_WARN(logger, ...) (void)0
#    define LOG_WARN(...) (void)0
#endif

#if GOB_LOG_ACTIVE_LEVEL <= GOB_LOG_LEVEL_ERROR
#    define GOB_LOGGER_ERROR(logger, ...) SPDLOG_LOGGER_CALL(logger, spdlog::level::err, __VA_ARGS__)
#    define LOG_ERROR(...) GOB_LOGGER_ERROR(gobot::Logger::GetInstance().GetLogger(), __VA_ARGS__)
#else
#    define SPDLOG_LOGGER_ERROR(logger, ...) (void)0
#    define LOG_ERROR(...) (void)0
#endif

#if GOB_LOG_ACTIVE_LEVEL <= GOB_LOG_LEVEL_CRITICAL
#    define GOB_LOGGER_CRITICAL(logger, ...) SPDLOG_LOGGER_CALL(logger, spdlog::level::critical, __VA_ARGS__)
#    define LOG_FATAL(...) GOB_LOGGER_CRITICAL(gobot::Logger::GetInstance().GetLogger(), __VA_ARGS__)
#else
#    define SPDLOG_LOGGER_CRITICAL(logger, ...) (void)0
#    define LOG_FATAL(...) (void)0
#endif


#define LOG_OFF gobot::Logger::GetInstance().GetLogger()->set_level(spdlog::level::off)
#define LOG_ON gobot::Logger::GetInstance().GetLogger()->set_level(spdlog::level::trace)
