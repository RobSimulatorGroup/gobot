/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-7-1
*/

#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include "fmt/format.h"

bool LineHasInclude(const std::string& lineBuffer, std::string& include_file_path) {
    std::string str = lineBuffer;
    // remove spaces
    str.erase(std::remove(str.begin(), str.end(), ' '), str.end());
    if (str.starts_with("//")) {
        return false;
    }
    if (str.starts_with("#include")) {
        // remove "#include"
        str.erase(0, 8);

        // remove ""
        str.erase(0, 1);
        str.erase(str.end()-1);
        include_file_path = str;
        return true;
    }
    return false;
}



// Return the source code of the complete shader
std::string Load(std::string path)
{
    std::string full_source_code;
    std::ifstream file(path);

    if (!file.is_open())
    {
        std::cerr << "ERROR: could not open the shader at: " << path << "\n" << std::endl;
        return full_source_code;
    }

    std::string line;
    while (std::getline(file, line))
    {
        // Look for the new shader include identifier
        std::string include_file_path;
        bool has_include = LineHasInclude(line, include_file_path);
        if (has_include)
        {
            // By using recursion, the new include file can be extracted
            // and inserted at this location in the shader source code
            std::filesystem::path shader_file_path = path;
            std::string cpp_path = (shader_file_path.parent_path() / include_file_path).string();
            full_source_code += Load(cpp_path);

            // Do not add this line to the shader source code, as the include
            // path would generate a compilation issue in the final source code
            continue;
        }

        full_source_code += line + '\n';
    }


    file.close();

    return full_source_code;
}



int main(int argc, char* argv[]) {
    std::string global_shader_var = argv[1];
    std::string cpp_file = argv[2];
    std::string shader_file = argv[3];

    std::ofstream out_file;
    std::ifstream in_file;

    out_file.open(cpp_file, std::ios::out);
    out_file << "/* Auto generated file, please don't modify. */ ";
//    out_file << "#include <string>  ";
    out_file << fmt::format("static const char* {} = R\"({})\";", global_shader_var, Load(shader_file));

    out_file.close();
    return 0;
}