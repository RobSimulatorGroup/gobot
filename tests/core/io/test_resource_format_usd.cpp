/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <gobot/core/io/resource_format_usd.hpp>
#include <gobot/scene/resources/packed_scene.hpp>

TEST(TestResourceFormatUSD, recognizes_usd_extensions_for_packed_scene) {
    gobot::Ref<gobot::ResourceFormatLoaderUSD> loader = gobot::MakeRef<gobot::ResourceFormatLoaderUSD>();

    std::vector<std::string> extensions;
    loader->GetRecognizedExtensionsForType("PackedScene", &extensions);

    EXPECT_NE(std::find(extensions.begin(), extensions.end(), "usd"), extensions.end());
    EXPECT_NE(std::find(extensions.begin(), extensions.end(), "usda"), extensions.end());
    EXPECT_NE(std::find(extensions.begin(), extensions.end(), "usdc"), extensions.end());
    EXPECT_TRUE(loader->HandlesType("PackedScene"));
}

TEST(TestResourceFormatUSD, disabled_openusd_loader_fails_without_crashing) {
    if (gobot::ResourceFormatLoaderUSD::IsOpenUSDAvailable()) {
        GTEST_SKIP() << "OpenUSD is enabled in this build.";
    }

    gobot::Ref<gobot::ResourceFormatLoaderUSD> loader = gobot::MakeRef<gobot::ResourceFormatLoaderUSD>();
    gobot::Ref<gobot::Resource> resource = loader->Load("res://missing.usda");
    EXPECT_FALSE(resource.IsValid());
}
