from conans import ConanFile, CMake

class GobotConan(ConanFile):
    name = "gobot"
    license = "LGPL-3.0"
    description = "Robot simulator"
    settings = "os", "compiler", "build_type", "arch"
    generators = "qt", "cmake", "cmake_find_package_multi", "cmake_paths"
    default_options = {"qt:shared": True, "gtest:shared": True}

    def requirements(self):
        self.requires("qt/5.15.7")
        self.requires("eigen/3.4.0")
        self.requires("spdlog/1.11.0")
        self.requires("nlohmann_json/3.11.2")
        self.requires("pybind11/2.10.0")
        self.requires("gtest/1.12.1")
        self.requires("magic_enum/0.8.1")
        self.requires("glad/0.1.36")
        self.requires("glfw/3.3.8")

    def configure(self):
        self.options['glad'].spec = 'gl'
        self.options['glad'].gl_profile = 'core'
        self.options["glad"].gl_version = "4.6"

    def imports(self):
        self.copy("*.dll", dst="bin", src="bin") # From bin to bin
        self.copy("*.dylib*", dst="bin", src="lib") # From lib to bin
