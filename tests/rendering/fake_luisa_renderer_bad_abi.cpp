#include "gobot/rendering/luisa_renderer_module_api.hpp"

extern "C" __attribute__((visibility("default")))
const gobot::LuisaRendererModuleApi* gobot_luisa_renderer_get_api() {
    static const gobot::LuisaRendererModuleApi api{
            gobot::GOBOT_LUISA_RENDERER_ABI_VERSION - 1u};
    return &api;
}
