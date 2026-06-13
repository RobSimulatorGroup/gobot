/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 2026-05-24
 * SPDX-License-Identifier: Apache-2.0
 */

#include <dlfcn.h>
#include <limits.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

#ifndef GOBOT_DEFAULT_PYTHON_LIBRARY
#define GOBOT_DEFAULT_PYTHON_LIBRARY ""
#endif

#ifndef GOBOT_DEFAULT_PYTHON_VERSION_MAJOR
#define GOBOT_DEFAULT_PYTHON_VERSION_MAJOR 0
#endif

#ifndef GOBOT_DEFAULT_PYTHON_VERSION_MINOR
#define GOBOT_DEFAULT_PYTHON_VERSION_MINOR 0
#endif

namespace {

std::filesystem::path GetExecutablePath() {
    std::string buffer(PATH_MAX, '\0');
    const ssize_t size = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
    if (size <= 0) {
        return {};
    }
    buffer.resize(static_cast<std::size_t>(size));
    return std::filesystem::path(buffer);
}

std::string DlErrorString() {
    const char* error = dlerror();
    return error != nullptr ? std::string(error) : std::string("unknown dynamic loader error");
}

bool ShouldPrintPythonLibrary() {
    const char* printed = std::getenv("GOBOT_PYTHON_LIBRARY_PRINTED");
    return printed == nullptr || std::string(printed) != "1";
}

bool IsInformationalInvocation(int argc, char* argv[]) {
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index] != nullptr ? argv[index] : "";
        if (argument == "--version" || argument == "-h" || argument == "--help") {
            return true;
        }
    }
    return false;
}

std::string ConfiguredPythonLibrary() {
    const char* python_library = std::getenv("GOBOT_PYTHON_LIBRARY");
    if (python_library != nullptr && !std::string(python_library).empty()) {
        return python_library;
    }
    return GOBOT_DEFAULT_PYTHON_LIBRARY;
}

std::vector<std::string> CandidatePythonLibraryNames() {
    std::vector<std::string> names;
    if constexpr (GOBOT_DEFAULT_PYTHON_VERSION_MAJOR > 0
            && GOBOT_DEFAULT_PYTHON_VERSION_MINOR > 0) {
        const std::string version = std::to_string(GOBOT_DEFAULT_PYTHON_VERSION_MAJOR)
                + "." + std::to_string(GOBOT_DEFAULT_PYTHON_VERSION_MINOR);
        names.push_back("libpython" + version + ".so.1.0");
        names.push_back("libpython" + version + ".so");
    }
    names.push_back("libpython3.so");
    return names;
}

std::string JoinStrings(const std::vector<std::string>& values) {
    std::string result;
    for (const std::string& value : values) {
        if (!result.empty()) {
            result += ", ";
        }
        result += value;
    }
    return result;
}

std::filesystem::path ResolvePythonLibraryPath(const std::string& configured_path) {
    const std::filesystem::path python_library_path(configured_path);
    std::error_code error;
    const bool is_directory = std::filesystem::is_directory(python_library_path, error);
    if (error || !is_directory) {
        return python_library_path;
    }

    for (const std::string& name : CandidatePythonLibraryNames()) {
        const std::filesystem::path candidate = python_library_path / name;
        std::error_code candidate_error;
        if (std::filesystem::is_regular_file(candidate, candidate_error)) {
            return candidate;
        }
    }

    return {};
}

} // namespace

int main(int argc, char* argv[]) {
    const std::string configured_python_library = ConfiguredPythonLibrary();
    if (configured_python_library.empty()) {
        std::cerr << "[gobot] GOBOT_PYTHON_LIBRARY is not set. "
                  << "Run gobot_editor through the Python package entry point "
                  << "or configure with -DGOBOT_PYTHON_LIBRARY=/path/to/libpython.so." << std::endl;
        return 127;
    }

    const std::filesystem::path python_library_path =
            ResolvePythonLibraryPath(configured_python_library);
    if (python_library_path.empty()) {
        std::cerr << "[gobot] GOBOT_PYTHON_LIBRARY points to directory '"
                  << configured_python_library
                  << "', but no libpython file was found there. Searched: "
                  << JoinStrings(CandidatePythonLibraryNames()) << std::endl;
        return 127;
    }
    const std::string python_library = python_library_path.string();

    if (ShouldPrintPythonLibrary() && !IsInformationalInvocation(argc, argv)) {
        std::cerr << "[gobot] Python library: " << python_library << std::endl;
    }

    void* python_handle = dlopen(python_library.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (python_handle == nullptr) {
        std::cerr << "[gobot] Failed to load Python library file '" << python_library
                  << "': " << DlErrorString() << std::endl;
        return 127;
    }

    const std::filesystem::path executable_path = GetExecutablePath();
    if (executable_path.empty()) {
        std::cerr << "[gobot] Failed to resolve gobot_editor executable path." << std::endl;
        return 127;
    }

    const std::filesystem::path runtime_path =
            executable_path.parent_path() / "libgobot_editor_runtime.so";
    void* runtime_handle = dlopen(runtime_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (runtime_handle == nullptr) {
        std::cerr << "[gobot] Failed to load editor runtime '" << runtime_path.string()
                  << "': " << DlErrorString() << std::endl;
        return 127;
    }

    dlerror();
    using EditorMain = int (*)(int, char**);
    auto* editor_main = reinterpret_cast<EditorMain>(dlsym(runtime_handle, "gobot_editor_main"));
    const char* symbol_error = dlerror();
    if (symbol_error != nullptr || editor_main == nullptr) {
        std::cerr << "[gobot] Failed to find gobot_editor_main in editor runtime: "
                  << (symbol_error != nullptr ? symbol_error : "symbol is null") << std::endl;
        return 127;
    }

    return editor_main(argc, argv);
}
