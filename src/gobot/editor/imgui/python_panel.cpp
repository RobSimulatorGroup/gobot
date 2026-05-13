/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/editor/imgui/python_panel.hpp"

#include <cstdlib>
#include <fstream>
#include <iterator>

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/io/python_script.hpp"
#include "gobot/core/io/resource_loader.hpp"
#include "gobot/core/io/resource.hpp"
#include "gobot/core/string_utils.hpp"
#include "gobot/editor/editor.hpp"
#include "gobot/editor/imgui/console_panel.hpp"
#include "gobot/editor/python_script_sync.hpp"
#include "gobot/log.hpp"
#include "gobot/python/python_script_runner.hpp"
#include "gobot/scene/node_3d.hpp"
#include "imgui.h"
#include "imgui_extension/icon_fonts/icons_material_design_icons.h"

namespace gobot {
namespace {

TextEditor::Palette PythonPalette() {
    TextEditor::Palette palette = TextEditor::GetDarkPalette();
    palette[static_cast<unsigned>(TextEditor::PaletteIndex::Default)] = IM_COL32(221, 224, 228, 255);
    palette[static_cast<unsigned>(TextEditor::PaletteIndex::Keyword)] = IM_COL32(125, 180, 255, 255);
    palette[static_cast<unsigned>(TextEditor::PaletteIndex::Number)] = IM_COL32(196, 214, 170, 255);
    palette[static_cast<unsigned>(TextEditor::PaletteIndex::String)] = IM_COL32(226, 169, 135, 255);
    palette[static_cast<unsigned>(TextEditor::PaletteIndex::CharLiteral)] = IM_COL32(226, 169, 135, 255);
    palette[static_cast<unsigned>(TextEditor::PaletteIndex::Punctuation)] = IM_COL32(205, 210, 218, 255);
    palette[static_cast<unsigned>(TextEditor::PaletteIndex::Identifier)] = IM_COL32(221, 224, 228, 255);
    palette[static_cast<unsigned>(TextEditor::PaletteIndex::KnownIdentifier)] = IM_COL32(113, 204, 183, 255);
    palette[static_cast<unsigned>(TextEditor::PaletteIndex::PreprocIdentifier)] = IM_COL32(205, 156, 220, 255);
    palette[static_cast<unsigned>(TextEditor::PaletteIndex::Comment)] = IM_COL32(135, 160, 130, 255);
    palette[static_cast<unsigned>(TextEditor::PaletteIndex::MultiLineComment)] = IM_COL32(135, 160, 130, 255);
    palette[static_cast<unsigned>(TextEditor::PaletteIndex::Background)] = IM_COL32(34, 36, 41, 255);
    palette[static_cast<unsigned>(TextEditor::PaletteIndex::Cursor)] = IM_COL32(235, 238, 242, 255);
    palette[static_cast<unsigned>(TextEditor::PaletteIndex::Selection)] = IM_COL32(75, 95, 125, 170);
    palette[static_cast<unsigned>(TextEditor::PaletteIndex::LineNumber)] = IM_COL32(118, 126, 140, 255);
    palette[static_cast<unsigned>(TextEditor::PaletteIndex::CurrentLineFill)] = IM_COL32(255, 255, 255, 18);
    palette[static_cast<unsigned>(TextEditor::PaletteIndex::CurrentLineFillInactive)] = IM_COL32(255, 255, 255, 10);
    palette[static_cast<unsigned>(TextEditor::PaletteIndex::CurrentLineEdge)] = IM_COL32(76, 83, 94, 255);
    return palette;
}

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
            "abs", "all", "any", "bool", "bytes", "callable", "chr", "dict", "dir",
            "enumerate", "Exception", "float", "getattr", "hasattr", "int", "isinstance",
            "len", "list", "max", "min", "object", "open", "print", "property", "range",
            "repr", "set", "setattr", "str", "sum", "super", "tuple", "type", "zip",
            "gobot"};
    for (const char* identifier_name : identifiers) {
        TextEditor::Identifier identifier;
        identifier.mDeclaration = "Python built-in";
        language.mIdentifiers.insert(std::make_pair(std::string(identifier_name), identifier));
    }

    language.mTokenRegexStrings.push_back(
            std::make_pair<std::string, TextEditor::PaletteIndex>("0[xX][0-9a-fA-F]+",
                                                                  TextEditor::PaletteIndex::Number));
    language.mTokenRegexStrings.push_back(
            std::make_pair<std::string, TextEditor::PaletteIndex>("0[bB][01]+",
                                                                  TextEditor::PaletteIndex::Number));
    language.mTokenRegexStrings.push_back(
            std::make_pair<std::string, TextEditor::PaletteIndex>(
                    "[+-]?(([0-9]+([.][0-9]*)?)|([.][0-9]+))([eE][+-]?[0-9]+)?",
                    TextEditor::PaletteIndex::Number));
    language.mTokenRegexStrings.push_back(
            std::make_pair<std::string, TextEditor::PaletteIndex>("[rRuUbBfF]*\\\"\\\"\\\".*\\\"\\\"\\\"",
                                                                  TextEditor::PaletteIndex::String));
    language.mTokenRegexStrings.push_back(
            std::make_pair<std::string, TextEditor::PaletteIndex>("[rRuUbBfF]*\\'\\'\\'.*\\'\\'\\'",
                                                                  TextEditor::PaletteIndex::String));
    language.mTokenRegexStrings.push_back(
            std::make_pair<std::string, TextEditor::PaletteIndex>("[rRuUbBfF]*\\\"(\\\\.|[^\\\"])*\\\"",
                                                                  TextEditor::PaletteIndex::String));
    language.mTokenRegexStrings.push_back(
            std::make_pair<std::string, TextEditor::PaletteIndex>("[rRuUbBfF]*\\'([^\\\\\\']|\\\\.)*\\'",
                                                                  TextEditor::PaletteIndex::String));
    language.mTokenRegexStrings.push_back(
            std::make_pair<std::string, TextEditor::PaletteIndex>("[a-zA-Z_][a-zA-Z0-9_]*",
                                                                  TextEditor::PaletteIndex::Identifier));
    language.mTokenRegexStrings.push_back(
            std::make_pair<std::string, TextEditor::PaletteIndex>("[\\[\\]\\{\\}\\!\\%\\^\\&\\*\\(\\)\\-\\+\\=\\~\\|\\<\\>\\?\\/\\:\\;\\,\\.]",
                                                                  TextEditor::PaletteIndex::Punctuation));

    language.mSingleLineComment = "#";
    language.mCommentStart = "\001";
    language.mCommentEnd = "\002";
    language.mPreprocChar = 0;
    language.mCaseSensitive = true;
    language.mAutoIndentation = true;
    language.mName = "Python";
    return language;
}

std::string DefaultPythonScript() {
    return "import gobot\n\n";
}

std::string NormalizePythonIndentation(std::string text) {
    return ReplaceAll(std::move(text), "\t", "    ");
}

void AddPythonMessage(const std::string& message,
                      ConsoleMessage::Level level,
                      const std::string& source) {
    ConsolePanel::AddMessage(MakeRef<ConsoleMessage>(message, level, source));
}

bool IsSaveShortcutPressed() {
    ImGuiIO& io = ImGui::GetIO();
    const bool ctrl = io.ConfigMacOSXBehaviors ? io.KeySuper : io.KeyCtrl;
    return ctrl && !io.KeyShift && !io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_S, false);
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
    editor_.SetPalette(PythonPalette());
    editor_.SetTabSize(4);
    editor_.SetShowWhitespaces(false);
    editor_.SetText(DefaultPythonScript());
}

bool PythonPanel::OpenScript(const std::string& path) {
    return LoadScript(path);
}

bool PythonPanel::LoadScript(const std::string& path) {
    const std::string local_path = ValidateLocalPath(path);
    const std::string global_path = ProjectSettings::GetInstance()->GlobalizePath(local_path);

    std::ifstream stream(global_path, std::ios::in);
    if (!stream.is_open()) {
        LOG_ERROR("Failed to open Python script '{}'.", local_path);
        return false;
    }

    const std::string source_code =
            NormalizePythonIndentation(std::string(std::istreambuf_iterator<char>(stream),
                                                   std::istreambuf_iterator<char>()));
    editor_.SetText(source_code);
    script_local_path_ = local_path;
    script_global_path_ = global_path;
    script_dirty_ = false;

    auto* editor = Editor::GetInstanceOrNull();
    SyncPythonScriptResourceSource(script_local_path_,
                                   source_code,
                                   editor != nullptr ? editor->GetEditedSceneRoot() : nullptr);
    return true;
}

void PythonPanel::OnImGuiContent() {
    const bool has_resource_path = !script_global_path_.empty();

    if (ImGui::Button(ICON_MDI_PLAY " Run Once")) {
        RunPythonScript();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Execute this Python script once against the active editor scene");
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
    const bool editor_focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);
    if (editor_focused && IsSaveShortcutPressed()) {
        SavePythonScript();
    }
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

    const std::string normalized_text = NormalizePythonIndentation(editor_.GetText());
    output << normalized_text;
    if (normalized_text != editor_.GetText()) {
        editor_.SetText(normalized_text);
    }
    auto* editor = Editor::GetInstanceOrNull();
    SyncPythonScriptResourceSource(script_local_path_,
                                   normalized_text,
                                   editor != nullptr ? editor->GetEditedSceneRoot() : nullptr);
    script_dirty_ = false;
    AddPythonMessage("Saved Python script: " + script_local_path_, ConsoleMessage::Info, "Python");

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
            python::PythonScriptRunner::ExecuteString(NormalizePythonIndentation(editor_.GetText()),
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
