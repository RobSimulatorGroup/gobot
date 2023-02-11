/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-10
*/


int main(int argc, char *argv[]) {

//    OS_LinuxBSD os;
//
//    setlocale(LC_CTYPE, "");
//
//    // We must override main when testing is enabled
//    TEST_MAIN_OVERRIDE
//
//    char *cwd = (char *)malloc(PATH_MAX);
//    ERR_FAIL_COND_V(!cwd, ERR_OUT_OF_MEMORY);
//    char *ret = getcwd(cwd, PATH_MAX);
//
//    Error err = Main::setup(argv[0], argc - 1, &argv[1]);
//    if (err != OK) {
//        free(cwd);
//
//        if (err == ERR_HELP) { // Returned by --help and --version, so success.
//            return 0;
//        }
//        return 255;
//    }
//
//    if (Main::start()) {
//        os.set_exit_code(EXIT_SUCCESS);
//        os.run(); // it is actually the OS that decides how to run
//    }
//    Main::cleanup();
//
//    if (ret) { // Previous getcwd was successful
//        if (chdir(cwd) != 0) {
//            ERR_PRINT("Couldn't return to previous working directory.");
//        }
//    }
//    free(cwd);
//
//    return os.get_exit_code();
}
