from conan import ConanFile
from conan.tools.cmake import cmake_layout

class GobotConan(ConanFile):
    name = "gobot"
    license = "LGPL-3.0"
    description = "Robot simulator"

    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def requirements(self):
        self.requires("eigen/3.4.0")
        self.requires("spdlog/1.11.0")
        self.requires("nlohmann_json/3.11.2")
        self.requires("pybind11/2.10.0")
        self.requires("magic_enum/0.9.5")
        self.requires("cxxopts/3.0.0")
        self.requires("assimp/5.2.2")
        self.requires("libpng/1.6.43")

    def layout(self):
        cmake_layout(self)

    def imports(self):
        self.copy("*.dll", dst="bin", src="bin") # From bin to bin
        self.copy("*.dylib*", dst="bin", src="lib") # From lib to bin
