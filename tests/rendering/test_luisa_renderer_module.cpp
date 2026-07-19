#include "gobot/rendering/luisa_renderer_module_api.hpp"

#include <dlfcn.h>

#include <gtest/gtest.h>

namespace gobot {

TEST(LuisaRendererModule, ExportsMatchingBackendNeutralAbi) {
    void* library = dlopen(GOBOT_TEST_LUISA_MODULE_PATH, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(library, nullptr) << dlerror();

    auto get_api = reinterpret_cast<GetLuisaRendererModuleApi>(
            dlsym(library, "gobot_luisa_renderer_get_api"));
    ASSERT_NE(get_api, nullptr) << dlerror();

    const LuisaRendererModuleApi* api = get_api();
    ASSERT_NE(api, nullptr);
    EXPECT_EQ(api->abi_version, GOBOT_LUISA_RENDERER_ABI_VERSION);
    ASSERT_NE(api->create, nullptr);
    ASSERT_NE(api->destroy, nullptr);
    ASSERT_NE(api->capabilities, nullptr);
    ASSERT_NE(api->render, nullptr);
    ASSERT_NE(api->reset_accumulation, nullptr);

    const SceneRendererCapabilities capabilities = api->capabilities(nullptr);
    EXPECT_TRUE(capabilities.ray_tracing_available);
    EXPECT_TRUE(capabilities.realtime);
    EXPECT_TRUE(capabilities.progressive);
    EXPECT_TRUE(capabilities.direct_presentation_interop);
    EXPECT_EQ(capabilities.backend_name, "LuisaCompute CUDA");

    EXPECT_EQ(dlclose(library), 0);
}

} // namespace gobot
