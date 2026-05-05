/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/editor/imgui/python_panel.hpp"

#include <cstdlib>
#include <fstream>
#include <iterator>

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/io/resource_loader.hpp"
#include "gobot/core/string_utils.hpp"
#include "gobot/editor/editor.hpp"
#include "gobot/editor/imgui/console_panel.hpp"
#include "gobot/log.hpp"
#include "gobot/python/python_script_runner.hpp"
#include "imgui.h"
#include "imgui_extension/icon_fonts/icons_material_design_icons.h"

namespace gobot {
namespace {

TextEditor::LanguageDefinition PythonLanguageDefinition() {
    TextEditor::LanguageDefinition language;

    static const char* const keywords[] = {
            "False", "None", "True", "and", "as", "assert", "async", "await",
            "break", "class", "continue", "def", "del", "elif", "else", "except",
            "finally", "for", "from", "global", "if", "import", "in", "is",
            "lambda", "nonlocal", "not", "or", "pass", "raise", "return",
            "try", "while", "with", "yield"};
    for (const char* keyword : keywords) {
        language.mKeywords.insert(keyword);
    }

    static const char* const identifiers[] = {
            "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
            "len", "list", "max", "min", "open", "print", "range", "set", "str",
            "sum", "tuple", "type", "zip", "gobot"};
    for (const char* identifier_name : identifiers) {
        TextEditor::Identifier identifier;
        identifier.mDeclaration = "Python built-in";
        language.mIdentifiers.insert(std::make_pair(std::string(identifier_name), identifier));
    }

    language.mTokenRegexStrings.push_back(
            std::make_pair<std::string, TextEditor::PaletteIndex>("\\\"\\\"\\\"(\\\\.|.|\\n)*?\\\"\\\"\\\"",
                                                                  TextEditor::PaletteIndex::String));
    language.mTokenRegexStrings.push_back(
            std::make_pair<std::string, TextEditor::PaletteIndex>("\\'\\'\\'(\\\\.|.|\\n)*?\\'\\'\\'",
                                                                  TextEditor::PaletteIndex::String));
    language.mTokenRegexStrings.push_back(
            std::make_pair<std::string, TextEditor::PaletteIndex>("[rRuUbBfF]*\\\"(\\\\.|[^\\\"])*\\\"",
                                                                  TextEditor::PaletteIndex::String));
    language.mTokenRegexStrings.push_back(
            std::make_pair<std::string, TextEditor::PaletteIndex>("[rRuUbBfF]*\\'([^\\\\\\']|\\\\.)*\\'",
                                                                  TextEditor::PaletteIndex::String));
    language.mTokenRegexStrings.push_back(
            std::make_pair<std::string, TextEditor::PaletteIndex>("[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?",
                                                                  TextEditor::PaletteIndex::Number));
    language.mTokenRegexStrings.push_back(
            std::make_pair<std::string, TextEditor::PaletteIndex>("0[xX][0-9a-fA-F]+",
                                                                  TextEditor::PaletteIndex::Number));
    language.mTokenRegexStrings.push_back(
            std::make_pair<std::string, TextEditor::PaletteIndex>("[a-zA-Z_][a-zA-Z0-9_]*",
                                                                  TextEditor::PaletteIndex::Identifier));
    language.mTokenRegexStrings.push_back(
            std::make_pair<std::string, TextEditor::PaletteIndex>("[\\[\\]\\{\\}\\!\\%\\^\\&\\*\\(\\)\\-\\+\\=\\~\\|\\<\\>\\?\\/\\:\\;\\,\\.]",
                                                                  TextEditor::PaletteIndex::Punctuation));

    language.mSingleLineComment = "#";
    language.mPreprocChar = 0;
    language.mCaseSensitive = true;
    language.mAutoIndentation = true;
    language.mName = "Python";
    return language;
}

std::string DefaultPythonScript() {
    return "import gobot\n"
           "\n"
           "ctx = gobot.app.context()\n"
           "print(ctx.root.name if ctx.root else 'no scene')\n";
}

void AddPythonMessage(const std::string& message,
                      ConsoleMessage::Level level,
                      const std::string& source) {
    ConsolePanel::AddMessage(MakeRef<ConsoleMessage>(message, level, source));
}

std::string ScratchScriptPath() {
    ProjectSettings* settings = ProjectSettings::GetInstance();
    if (settings == nullptr || settings->GetProjectPath().empty()) {
        return {};
    }

    std::string candidate = "res://scripts/script.py";
    for (int index = 1; ResourceLoader::Exists(candidate, "PythonScript"); ++index) {
        candidate = "res://scripts/script_" + std::to_string(index) + ".py";
    }
    return candidate;
}

} // namespace

PythonPanel::PythonPanel() {
    SetName("PythonPanel");
    SetImGuiWindow(ICON_MDI_LANGUAGE_PYTHON " Python", "python");
    SetImGuiWindowSize(Vector2f(760, 520), ImGuiCond_FirstUseEver);

    editor_.SetLanguageDefinition(PythonLanguageDefinition());
    editor_.SetTabSize(4);
    editor_.SetShowWhitespaces(false);
    editor_.SetText(DefaultPythonScript());
}

bool PythonPanel::OpenScript(const std::string& path) {
    const std::string local_path = ValidateLocalPath(path);
    const std::string global_path = ProjectSettings::GetInstance()->GlobalizePath(local_path);

    std::ifstream stream(global_path, std::ios::in);
    if (!stream.is_open()) {
        LOG_ERROR("Failed to open Python script '{}'.", local_path);
        return false;
    }

    editor_.SetText(std::string(std::istreambuf_iterator<char>(stream),
                                std::istreambuf_iterator<char>()));
    script_local_path_ = local_path;
    script_global_path_ = global_path;
    script_dirty_ = false;
    return true;
}

void PythonPanel::OnImGuiContent() {
    const bool has_resource_path = !script_global_path_.empty();

    if (ImGui::Button(ICON_MDI_PLAY " Run")) {
        RunPythonScript();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Run Python against the active editor scene");
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_MDI_CONTENT_SAVE " Save")) {
        SavePythonScript();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(has_resource_path
                          ? "Save the current Python resource"
                          : "Create a Python resource from this scratch script");
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_MDI_OPEN_IN_NEW " VS Code")) {
        OpenInVSCode();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(has_resource_path
                          ? "Open the current Python resource in VS Code"
                          : "Create a Python resource and open it in VS Code");
    }

    ImGui::SameLine();
    if (has_resource_path) {
        ImGui::TextUnformatted(script_dirty_
                               ? (script_local_path_ + " *").c_str()
                               : script_local_path_.c_str());
    } else {
        ImGui::TextDisabled("[scratch]");
    }

    ImGui::Separator();

    editor_.Render("##PythonScriptEditor", ImGui::GetContentRegionAvail(), false);
    if (editor_.IsTextChanged()) {
        script_dirty_ = true;
    }
}

bool PythonPanel::SavePythonScript() {
    if (script_global_path_.empty()) {
        script_local_path_ = ScratchScriptPath();
        if (script_local_path_.empty()) {
            AddPythonMessage("Open or create a project before saving Python scripts.",
                             ConsoleMessage::Warn,
                             "Python");
            return false;
        }
        script_global_path_ = ProjectSettings::GetInstance()->GlobalizePath(script_local_path_);

        std::error_code error;
        std::filesystem::create_directories(std::filesystem::path(script_global_path_).parent_path(), error);
        if (error) {
            LOG_ERROR("Failed to create Python script directory '{}': {}",
                      std::filesystem::path(script_global_path_).parent_path().string(),
                      error.message());
            script_local_path_.clear();
            script_global_path_.clear();
            return false;
        }
    }

    std::ofstream output(script_global_path_, std::ios::out | std::ios::trunc);
    if (!output.is_open()) {
        LOG_ERROR("Failed to save Python script '{}'.", script_local_path_);
        return false;
    }

    output << editor_.GetText();
    script_dirty_ = false;
    AddPythonMessage("Saved Python script: " + script_local_path_, ConsoleMessage::Info, "Python");

    auto* editor = Editor::GetInstanceOrNull();
    if (editor != nullptr) {
        editor->RefreshResourcePanel();
    }
    return true;
}

bool PythonPanel::OpenInVSCode() {
    if (script_global_path_.empty()) {
        if (!SavePythonScript()) {
            return false;
        }
    }

    const std::string command = "code --reuse-window \"" + script_global_path_ + "\"";
    const int result = std::system(command.c_str());
    if (result != 0) {
        LOG_ERROR("Failed to open '{}' in VS Code with command '{}'.", script_global_path_, command);
        return false;
    }
    return true;
}

void PythonPanel::RunPythonScript() {
    auto* editor = Editor::GetInstanceOrNull();
    auto* context = editor != nullptr ? editor->GetEngineContext() : nullptr;
    python::PythonExecutionResult result =
            python::PythonScriptRunner::ExecuteString(editor_.GetText(),
                                                      context,
                                                      script_local_path_.empty()
                                                              ? "<editor-python-panel>"
                                                              : script_local_path_);

    if (!result.output.empty()) {
        AddPythonMessage(result.output, ConsoleMessage::Info, "Python stdout");
    }
    if (!result.error.empty()) {
        AddPythonMessage(result.error,
                         result.ok ? ConsoleMessage::Warn : ConsoleMessage::Error,
                         "Python stderr");
    }
    if (result.ok && result.output.empty() && result.error.empty()) {
        AddPythonMessage("Python script executed.", ConsoleMessage::Info, "Python");
    }
}

} // namespace gobot
