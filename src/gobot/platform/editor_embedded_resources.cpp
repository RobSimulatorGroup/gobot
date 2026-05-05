/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/platform/editor_embedded_resources.hpp"

#include "gobot/generated/editor_icon_svg.hpp"

namespace gobot {

const char* GetEmbeddedEditorIconSvg() {
    return reinterpret_cast<const char*>(icon_svg);
}

std::size_t GetEmbeddedEditorIconSvgSize() {
    return static_cast<std::size_t>(icon_svg_len);
}

} // namespace gobot
