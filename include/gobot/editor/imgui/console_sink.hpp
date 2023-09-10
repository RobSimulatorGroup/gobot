/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-10
*/

#pragma once

#include "gobot/editor/imgui/console_panel.hpp"
#include <spdlog/sinks/base_sink.h>
#include <mutex>

namespace gobot {

template <typename Mutex>
class ImGuiConsoleSink : public spdlog::sinks::base_sink<Mutex>
{
public:
    explicit ImGuiConsoleSink() {};

    ImGuiConsoleSink(const ImGuiConsoleSink&)            = delete;
    ImGuiConsoleSink& operator=(const ImGuiConsoleSink&) = delete;
    virtual ~ImGuiConsoleSink()                          = default;

    // SPDLog sink interface
    void sink_it_(const spdlog::details::log_msg& msg) override
    {
        spdlog::memory_buf_t formatted;
        spdlog::sinks::base_sink<Mutex>::formatter_->format(msg, formatted);
        std::string source = fmt::format("File : {0} | Function : {1} | Line : {2}", msg.source.filename, msg.source.funcname, msg.source.line);
        auto message = MakeRef<ConsoleMessage>(fmt::to_string(formatted),
                                               GetMessageLevel(msg.level),
                                               source,
                                               static_cast<int>(msg.thread_id));
        ConsolePanel::AddMessage(message);
    }

    static ConsoleMessage::Level GetMessageLevel(const spdlog::level::level_enum level)
    {
        switch(level)
        {
            case spdlog::level::level_enum::trace:
                return ConsoleMessage::Level::Trace;
            case spdlog::level::level_enum::debug:
                return ConsoleMessage::Level::Debug;
            case spdlog::level::level_enum::info:
                return ConsoleMessage::Level::Info;
            case spdlog::level::level_enum::warn:
                return ConsoleMessage::Level::Warn;
            case spdlog::level::level_enum::err:
                return ConsoleMessage::Level::Error;
            case spdlog::level::level_enum::critical:
                return ConsoleMessage::Level::Critical;
        }
        return ConsoleMessage::Level::Trace;
    }

    void flush_() override
    {
        ConsolePanel::Flush();
    };
};

using ImGuiConsoleSinkMultiThreaded = ImGuiConsoleSink<std::mutex>;  // multi-thread

}

