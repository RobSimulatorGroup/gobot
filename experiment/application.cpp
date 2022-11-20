/* Copyright(c) 2020-2022, Qiqi Wu<1258552199@qq.com>.
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
   The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

#include "gobot/main/application.hpp"
#include "gobot/log.hpp"

#include <set>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace gobot::main {
const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

static const std::vector<const char *> validation_layers = {
        "VK_LAYER_KHRONOS_validation"
};



static VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageType,
              const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
              void* pUserData) {

    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
}


void Application::run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
}


void Application::initWindow() {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window_ = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
}

void Application::mainLoop() {
    while (!glfwWindowShouldClose(window_)) {
        glfwPollEvents();
    }
}

void Application::cleanup() {
    device_.destroy();
    instance_.destroySurfaceKHR(surface_, nullptr);
    instance_.destroy();

    glfwDestroyWindow(window_);
    glfwTerminate();
}

bool Application::checkValidationLayerSupport() {
    auto layers = vk::enumerateInstanceLayerProperties();

    for (const char *layer_name: validation_layers) {
        bool layerFound = false;
        for (const auto &layer: layers) {
            if (strcmp(layer_name, layer.layerName) == 0) {
                layerFound = true;
                break;
            }
        }
        if (!layerFound) {
            return false;
        }
    }

    return true;
}

void Application::pickPhysicalDevice() {
    auto devices = instance_.enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }
    for (const auto& device: devices) {
        if (isDeviceSuitable(device)) {
            physical_device_ = device;
            break;
        }
    }
    if (!physical_device_) {
        throw std::runtime_error("failed to find a suitable GPU!");
    }
}


Application::QueueFamilyIndices Application::findQueueFamilies(const vk::PhysicalDevice& device) {
    QueueFamilyIndices indices;
    int i = 0;
    auto queue_families = device.getQueueFamilyProperties();
    for (const auto& queue_family : queue_families) {
        if (queue_family.queueFlags & vk::QueueFlagBits::eGraphics) {
            indices.graphics_family = i;
        }
        if (device.getSurfaceSupportKHR(i, surface_) ) {
            indices.present_family = i;
        }

        if (indices.isComplete()) {
            break;
        }
        i++;
    }
    return indices;
}

bool Application::isDeviceSuitable(const vk::PhysicalDevice& device) {
    QueueFamilyIndices indices = findQueueFamilies(device);

    return indices.isComplete();
}

std::vector<const char*> Application::getRequiredExtensions() {
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    extensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    return extensions;
}

void Application::setupDebugMessenger() {

    vk::DebugUtilsMessengerCreateInfoEXT debug_utils_messenger_create_info;
    debug_utils_messenger_create_info.setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                                             vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                                             vk::DebugUtilsMessageSeverityFlagBitsEXT::eError)
                          .setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                                                  vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                                                  vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation)
                                                  .setPfnUserCallback(debugCallback);
    auto dldy = vk::DispatchLoaderDynamic(instance_, vkGetInstanceProcAddr);
    auto debug_report = instance_.createDebugUtilsMessengerEXTUnique(debug_utils_messenger_create_info,
                                                                                    nullptr,
                                                                                    dldy);

    if (!debug_report) {
        throw std::runtime_error("failed to set up debug messenger!");
    }
}


void Application::createInstance() {
    if (!checkValidationLayerSupport()) {
        throw std::runtime_error("validation layers requested, but not available!");
    }


    vk::ApplicationInfo appInfo;
    appInfo.setApiVersion(VK_MAKE_VERSION(1, 0, 0))
            .setApplicationVersion(VK_MAKE_VERSION(1, 0, 0))
            .setPEngineName("No Engine")
            .setEngineVersion(VK_MAKE_VERSION(1, 0, 0))
            .setApiVersion(VK_MAKE_VERSION(1, 0, 0))
            .setPApplicationName("Robot Simulator");


    vk::InstanceCreateInfo createInfo;
    auto required_extensions = Application::getRequiredExtensions();
    createInfo.setPApplicationInfo(&appInfo)
              .setPEnabledExtensionNames(required_extensions)
              .setPEnabledLayerNames(validation_layers);


    instance_ = vk::createInstance(createInfo);
    if (!instance_) {
        throw std::runtime_error("Create Vk instance failed");
    }
}

void Application::createSurface() {
    VkSurfaceKHR surfaceTmp{};

    if (glfwCreateWindowSurface(instance_, window_, nullptr, &surfaceTmp) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }

    surface_ = vk::SurfaceKHR(surfaceTmp);
}

void Application::createLogicalDevice() {
    QueueFamilyIndices indices = findQueueFamilies(physical_device_);

    std::vector<vk::DeviceQueueCreateInfo> queue_create_infos;
    std::set<uint32_t> unique_queue_families{indices.graphics_family.value(), indices.present_family.value()};
    float queue_priority = 1.0;
    for (const auto& queue_family : unique_queue_families) {
        vk::DeviceQueueCreateInfo device_queue_create_info;
        device_queue_create_info.setQueueFamilyIndex(queue_family)
                                .setQueueCount(1)
                                .setPQueuePriorities(&queue_priority);
        queue_create_infos.emplace_back(device_queue_create_info);
    }


    vk::PhysicalDeviceFeatures device_features;
    vk::DeviceCreateInfo device_create_info;
    device_create_info.setQueueCreateInfos(queue_create_infos)
                      .setPEnabledFeatures(&device_features)
                      .setPEnabledLayerNames(validation_layers);

    device_ = physical_device_.createDevice(device_create_info, nullptr);
    if (!device_) {
        throw std::runtime_error("failed to create logical device!");
    }

    graphics_queue_ = device_.getQueue( indices.graphics_family.value(), 0);
    present_queue_ = device_.getQueue( indices.present_family.value(), 0);
}

void Application::initVulkan() {
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
}

}