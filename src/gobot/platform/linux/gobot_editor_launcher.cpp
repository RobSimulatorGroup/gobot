/*
 * The gobot is a robot simulation platform.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <dlfcn.h>
#include <limits.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

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

} // namespace

int main(int argc, char* argv[]) {
    const char* python_library = std::getenv("GOBOT_PYTHON_LIBRARY");
    if (python_library == nullptr || std::string(python_library).empty()) {
        std::cerr << "[gobot] GOBOT_PYTHON_LIBRARY is not set. "
                  << "Run gobot_editor through the Python package entry point "
                  << "or set it to the current environment's libpython path." << std::endl;
        return 127;
    }

    if (ShouldPrintPythonLibrary()) {
        std::cerr << "[gobot] Python library: " << python_library << std::endl;
    }

    void* python_handle = dlopen(python_library, RTLD_NOW | RTLD_GLOBAL);
    if (python_handle == nullptr) {
        std::cerr << "[gobot] Failed to load Python library '" << python_library
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
