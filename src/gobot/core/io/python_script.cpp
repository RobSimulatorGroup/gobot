/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/core/io/python_script.hpp"

#include <fstream>
#include <iterator>

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/core/string_utils.hpp"
#include "gobot/error_macros.hpp"

namespace gobot {

void PythonScript::SetSourceCode(const std::string& source_code) {
    source_code_ = source_code;
}

const std::string& PythonScript::GetSourceCode() const {
    return source_code_;
}

Ref<Resource> ResourceFormatLoaderPythonScript::Load(const std::string& path,
                                                     const std::string& original_path,
                                                     CacheMode cache_mode) {
    const std::string global_path =
            ProjectSettings::GetInstance()->GlobalizePath(ValidateLocalPath(path));
    ERR_FAIL_COND_V_MSG(!std::filesystem::exists(global_path),
                        {},
                        fmt::format("Cannot open Python script: {}.", path));

    std::ifstream stream(global_path, std::ios::in);
    ERR_FAIL_COND_V_MSG(!stream.good(),
                        {},
                        fmt::format("Cannot read Python script: {}.", path));

    auto script = MakeRef<PythonScript>();
    script->SetSourceCode(std::string(std::istreambuf_iterator<char>(stream),
                                      std::istreambuf_iterator<char>()));
    return script;
}

void ResourceFormatLoaderPythonScript::GetRecognizedExtensions(std::vector<std::string>* extensions) const {
    extensions->push_back("py");
}

bool ResourceFormatLoaderPythonScript::HandlesType(const std::string& type) const {
    return type.empty() || type == "PythonScript" || type == "Script" || type == "Resource";
}

} // namespace gobot

GOBOT_REGISTRATION {
    Class_<PythonScript>("PythonScript")
            .constructor()(CtorAsRawPtr)
            .property("source_code", &PythonScript::GetSourceCode, &PythonScript::SetSourceCode)(
                    AddMetaPropertyInfo(
                            PropertyInfo()
                                .SetUsageFlags(PropertyUsageFlags::Storage)));
    gobot::Type::register_wrapper_converter_for_base_classes<Ref<PythonScript>, Ref<Resource>>();
}
