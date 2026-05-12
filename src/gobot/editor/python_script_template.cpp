#include "gobot/editor/python_script_template.hpp"

namespace gobot {

std::string NodeScriptTemplate() {
    return R"(import gobot


class Script(gobot.NodeScript):
    def _ready(self):
        pass

    def _process(self, delta: float):
        pass

    def _physics_process(self, delta: float):
        pass
)";
}

std::string ToolScriptTemplate() {
    return R"(import gobot

ctx = gobot.app.context()
root = ctx.root

print(root.name if root else "no scene")
)";
}

} // namespace gobot
