#pragma once

#include <cstdint>
#include <string>

extern "C" {

bool gobot_task_llvm_available();
const char* gobot_task_llvm_version();
const char* gobot_task_llvm_last_error();
int gobot_task_llvm_compile_cpp(const char* source,
                                const char* virtual_path,
                                const char* object_path,
                                int optimization_level,
                                bool fast_math,
                                bool debug);
int gobot_task_llvm_load_obj(const char* object_path, const char* module_name);
std::uint64_t gobot_task_llvm_lookup(const char* module_name, const char* symbol_name);
int gobot_task_llvm_unload(const char* module_name);

}

namespace gobot::rl {

struct TaskLlvmCompileOptions {
    int optimization_level{3};
    bool fast_math{true};
    bool debug{false};
};

inline bool TaskLlvmAvailable() {
    return gobot_task_llvm_available();
}

inline std::string TaskLlvmVersion() {
    return gobot_task_llvm_version();
}

inline std::string TaskLlvmLastError() {
    return gobot_task_llvm_last_error();
}

inline int TaskLlvmCompileCpp(const std::string& source,
                              const std::string& virtual_path,
                              const std::string& object_path,
                              const TaskLlvmCompileOptions& options = {}) {
    return gobot_task_llvm_compile_cpp(source.c_str(),
                                       virtual_path.c_str(),
                                       object_path.c_str(),
                                       options.optimization_level,
                                       options.fast_math,
                                       options.debug);
}

inline int TaskLlvmLoadObject(const std::string& object_path, const std::string& module_name) {
    return gobot_task_llvm_load_obj(object_path.c_str(), module_name.c_str());
}

inline std::uint64_t TaskLlvmLookup(const std::string& module_name, const std::string& symbol_name) {
    return gobot_task_llvm_lookup(module_name.c_str(), symbol_name.c_str());
}

inline int TaskLlvmUnload(const std::string& module_name) {
    return gobot_task_llvm_unload(module_name.c_str());
}

} // namespace gobot::rl
