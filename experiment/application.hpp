/* Copyright(c) 2020-2022, Qiqi Wu<1258552199@qq.com>.
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * This file is created by Qiqi Wu, 22-9-24
*/

#pragma once

#include "vulkan/vulkan.hpp"

class GLFWwindow;

namespace gobot::main {

class Application {
public:

    void run();

private:
    void initVulkan();

    void initWindow();

    void pickPhysicalDevice();

    void mainLoop();

    void cleanup();

    void createInstance();

    void createLogicalDevice();

    static bool checkValidationLayerSupport();

    static std::vector<const char *> getRequiredExtensions();

private:
    struct QueueFamilyIndices {
        std::optional<uint32_t> graphics_family;
        std::optional<uint32_t> present_family;

        [[nodiscard]] bool isComplete() const {
            return graphics_family.has_value() &&  present_family.has_value();
        }
    };

    void setupDebugMessenger();

    bool isDeviceSuitable(const vk::PhysicalDevice& device);

    QueueFamilyIndices findQueueFamilies(const vk::PhysicalDevice& device);

    void createSurface();

private:

    GLFWwindow *window_{nullptr};


    vk::Instance instance_;
    vk::PhysicalDevice physical_device_;
    vk::Device device_;
    vk::Queue graphics_queue_;
    vk::Queue present_queue_;
    vk::SurfaceKHR surface_;
};


}
