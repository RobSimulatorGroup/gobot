#include "task_llvm.hpp"

#include <mutex>
#include <sstream>
#include <system_error>
#include <unordered_map>

#ifdef GOBOT_HAS_TASK_LLVM
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Lex/PreprocessorOptions.h>

#include <llvm/Config/llvm-config.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#if LLVM_VERSION_MAJOR >= 16
#include <llvm/TargetParser/Host.h>
#else
#include <llvm/Support/Host.h>
#endif
#endif

namespace {

std::mutex g_task_llvm_mutex;
std::string g_task_llvm_last_error;

void SetLastError(std::string message) {
    g_task_llvm_last_error = std::move(message);
}

#ifdef GOBOT_HAS_TASK_LLVM

llvm::orc::LLJIT* g_task_llvm_jit = nullptr;
std::unordered_map<std::string, llvm::orc::JITDylib*> g_task_llvm_modules;

void InitializeLlvm() {
    static const bool initialized = []() {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();
        return true;
    }();
    (void)initialized;
}

std::unique_ptr<clang::CompilerInstance> CreateCompiler(const std::string& virtual_path,
                                                        const gobot::rl::TaskLlvmCompileOptions& options) {
    InitializeLlvm();
    auto compiler = std::make_unique<clang::CompilerInstance>();

    std::vector<const char*> args;
    args.push_back(virtual_path.c_str());
    args.push_back("-std=c++17");
    args.push_back("-triple");
    const std::string target_triple = llvm::sys::getDefaultTargetTriple();
    args.push_back(target_triple.c_str());
    if (options.fast_math) {
        args.push_back("-ffast-math");
    }
    if (options.debug) {
        args.push_back("-O0");
    } else {
        switch (options.optimization_level) {
            case 0:
                args.push_back("-O0");
                break;
            case 1:
                args.push_back("-O1");
                break;
            case 2:
                args.push_back("-O2");
                break;
            default:
                args.push_back("-O3");
                break;
        }
    }

#if LLVM_VERSION_MAJOR >= 21
    clang::DiagnosticOptions diagnostic_options;
    auto text_diagnostic_printer = std::make_unique<clang::TextDiagnosticPrinter>(llvm::errs(), diagnostic_options);
    clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagnostic_ids;
    auto diagnostics =
            std::make_unique<clang::DiagnosticsEngine>(diagnostic_ids, diagnostic_options, text_diagnostic_printer.release());
#else
    clang::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagnostic_options = new clang::DiagnosticOptions();
    auto text_diagnostic_printer = std::make_unique<clang::TextDiagnosticPrinter>(llvm::errs(), &*diagnostic_options);
    clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagnostic_ids;
    auto diagnostics = std::make_unique<clang::DiagnosticsEngine>(diagnostic_ids,
                                                                  &*diagnostic_options,
                                                                  text_diagnostic_printer.release());
#endif

    clang::CompilerInvocation::CreateFromArgs(compiler->getInvocation(), args, *diagnostics);
    if (!options.debug) {
        compiler->getPreprocessorOpts().addMacroDef("NDEBUG");
    }
    compiler->getLangOpts().MicrosoftExt = 1;
    compiler->getLangOpts().DeclSpecKeyword = 1;

#if LLVM_VERSION_MAJOR >= 21
    diagnostics->setClient(diagnostics->getClient(), false);
#endif
#if LLVM_VERSION_MAJOR >= 22
    compiler->createDiagnostics(diagnostics->getClient(), true);
#elif LLVM_VERSION_MAJOR == 21
    compiler->createDiagnostics(*llvm::vfs::getRealFileSystem(), diagnostics->getClient(), true);
#else
    compiler->createDiagnostics(text_diagnostic_printer.get(), false);
#endif
    return compiler;
}

std::unique_ptr<llvm::Module> SourceToLlvmModule(const std::string& source,
                                                 const std::string& virtual_path,
                                                 llvm::LLVMContext& context,
                                                 const gobot::rl::TaskLlvmCompileOptions& options) {
    auto compiler = CreateCompiler(virtual_path, options);
    std::unique_ptr<llvm::MemoryBuffer> buffer = llvm::MemoryBuffer::getMemBufferCopy(source);
    compiler->getInvocation().getPreprocessorOpts().addRemappedFile(virtual_path.c_str(), buffer.get());
    clang::EmitLLVMOnlyAction action(&context);
    const bool success = compiler->ExecuteAction(action);
    (void)buffer.release();
    return success ? std::move(action.takeModule()) : nullptr;
}

llvm::orc::LLJIT* GetOrCreateJit() {
    InitializeLlvm();
    if (g_task_llvm_jit != nullptr) {
        return g_task_llvm_jit;
    }
    llvm::orc::LLJITBuilder builder;
    auto jit = builder.create();
    if (!jit) {
        SetLastError(llvm::toString(jit.takeError()));
        return nullptr;
    }
    g_task_llvm_jit = (*jit).release();
    return g_task_llvm_jit;
}

#endif

} // namespace

extern "C" bool gobot_task_llvm_available() {
#ifdef GOBOT_HAS_TASK_LLVM
    return true;
#else
    return false;
#endif
}

extern "C" const char* gobot_task_llvm_version() {
#ifdef GOBOT_HAS_TASK_LLVM
    return LLVM_VERSION_STRING;
#else
    return "unavailable";
#endif
}

extern "C" const char* gobot_task_llvm_last_error() {
    std::lock_guard<std::mutex> lock(g_task_llvm_mutex);
    return g_task_llvm_last_error.c_str();
}

extern "C" int gobot_task_llvm_compile_cpp(const char* source,
                                           const char* virtual_path,
                                           const char* object_path,
                                           int optimization_level,
                                           bool fast_math,
                                           bool debug) {
#ifndef GOBOT_HAS_TASK_LLVM
    (void)source;
    (void)virtual_path;
    (void)object_path;
    (void)optimization_level;
    (void)fast_math;
    (void)debug;
    std::lock_guard<std::mutex> lock(g_task_llvm_mutex);
    SetLastError("Gobot was built without task LLVM support");
    return -1;
#else
    std::lock_guard<std::mutex> lock(g_task_llvm_mutex);
    SetLastError("");
    gobot::rl::TaskLlvmCompileOptions options;
    options.optimization_level = optimization_level;
    options.fast_math = fast_math;
    options.debug = debug;
    llvm::LLVMContext context;
    std::unique_ptr<llvm::Module> module = SourceToLlvmModule(source != nullptr ? source : "",
                                                              virtual_path != nullptr ? virtual_path : "task_kernel.cpp",
                                                              context,
                                                              options);
    if (!module) {
        SetLastError("failed to lower task kernel C++ source to LLVM IR");
        return -1;
    }

    std::string error;
    const std::string target_triple = llvm::sys::getDefaultTargetTriple();
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(target_triple, error);
    if (target == nullptr) {
        SetLastError(error.empty() ? "failed to resolve LLVM target" : error);
        return -2;
    }

    llvm::TargetOptions target_options;
    target_options.AllowFPOpFusion = options.fast_math ? llvm::FPOpFusion::Fast : llvm::FPOpFusion::Standard;
    llvm::Reloc::Model relocation_model = llvm::Reloc::PIC_;
    llvm::CodeModel::Model code_model = llvm::CodeModel::Large;
    llvm::CodeGenOptLevel codegen_opt = llvm::CodeGenOptLevel::Aggressive;
    if (options.debug || options.optimization_level <= 0) {
        codegen_opt = llvm::CodeGenOptLevel::None;
    } else if (options.optimization_level == 1) {
        codegen_opt = llvm::CodeGenOptLevel::Less;
    } else if (options.optimization_level == 2) {
        codegen_opt = llvm::CodeGenOptLevel::Default;
    }

    const std::string cpu = llvm::sys::getHostCPUName().str();
    llvm::StringMap<bool> feature_map;
    llvm::sys::getHostCPUFeatures(feature_map);
    std::string features;
    for (const auto& feature : feature_map) {
        if (!features.empty()) {
            features += ",";
        }
        features += feature.second ? "+" : "-";
        features += feature.first().str();
    }

    std::unique_ptr<llvm::TargetMachine> target_machine(
            target->createTargetMachine(target_triple,
                                        cpu.empty() ? "generic" : cpu,
                                        features,
                                        target_options,
                                        relocation_model,
                                        code_model,
                                        codegen_opt));
    if (!target_machine) {
        SetLastError("failed to create LLVM target machine");
        return -3;
    }
    module->setDataLayout(target_machine->createDataLayout());
    module->setTargetTriple(target_triple);

    std::error_code error_code;
    llvm::raw_fd_ostream output(object_path != nullptr ? object_path : "", error_code, llvm::sys::fs::OF_None);
    if (error_code) {
        SetLastError(error_code.message());
        return -4;
    }

    llvm::legacy::PassManager pass_manager;
    if (target_machine->addPassesToEmitFile(pass_manager, output, nullptr, llvm::CodeGenFileType::ObjectFile)) {
        SetLastError("target machine cannot emit object files");
        return -5;
    }
    pass_manager.run(*module);
    output.flush();
    if (output.has_error()) {
        SetLastError("failed to write object file");
        return -6;
    }
    return 0;
#endif
}

extern "C" int gobot_task_llvm_load_obj(const char* object_path, const char* module_name) {
#ifndef GOBOT_HAS_TASK_LLVM
    (void)object_path;
    (void)module_name;
    std::lock_guard<std::mutex> lock(g_task_llvm_mutex);
    SetLastError("Gobot was built without task LLVM support");
    return -1;
#else
    std::lock_guard<std::mutex> lock(g_task_llvm_mutex);
    SetLastError("");
    auto* jit = GetOrCreateJit();
    if (jit == nullptr) {
        return -1;
    }
    const std::string module_key = module_name != nullptr ? module_name : "";
    if (g_task_llvm_modules.find(module_key) != g_task_llvm_modules.end()) {
        return 0;
    }
    auto dylib = jit->createJITDylib(module_key);
    if (!dylib) {
        SetLastError(llvm::toString(dylib.takeError()));
        return -2;
    }
    auto buffer = llvm::MemoryBuffer::getFile(object_path != nullptr ? object_path : "");
    if (!buffer) {
        SetLastError(buffer.getError().message());
        return -3;
    }
    llvm::orc::JITDylib& jit_dylib = *dylib;
    if (auto error = jit->addObjectFile(jit_dylib, std::move(*buffer))) {
        SetLastError(llvm::toString(std::move(error)));
        return -4;
    }
    g_task_llvm_modules[module_key] = &jit_dylib;
    return 0;
#endif
}

extern "C" std::uint64_t gobot_task_llvm_lookup(const char* module_name, const char* symbol_name) {
#ifndef GOBOT_HAS_TASK_LLVM
    (void)module_name;
    (void)symbol_name;
    std::lock_guard<std::mutex> lock(g_task_llvm_mutex);
    SetLastError("Gobot was built without task LLVM support");
    return 0;
#else
    std::lock_guard<std::mutex> lock(g_task_llvm_mutex);
    SetLastError("");
    auto* jit = GetOrCreateJit();
    if (jit == nullptr) {
        return 0;
    }
    const std::string module_key = module_name != nullptr ? module_name : "";
    const auto found = g_task_llvm_modules.find(module_key);
    if (found == g_task_llvm_modules.end()) {
        SetLastError("task LLVM module is not loaded: " + module_key);
        return 0;
    }
    auto symbol = jit->lookup(*found->second, symbol_name != nullptr ? symbol_name : "");
    if (!symbol) {
        SetLastError(llvm::toString(symbol.takeError()));
        return 0;
    }
    return symbol->getValue();
#endif
}

extern "C" int gobot_task_llvm_unload(const char* module_name) {
#ifndef GOBOT_HAS_TASK_LLVM
    (void)module_name;
    std::lock_guard<std::mutex> lock(g_task_llvm_mutex);
    SetLastError("Gobot was built without task LLVM support");
    return -1;
#else
    std::lock_guard<std::mutex> lock(g_task_llvm_mutex);
    SetLastError("");
    if (g_task_llvm_jit == nullptr) {
        return 0;
    }
    const std::string module_key = module_name != nullptr ? module_name : "";
    auto found = g_task_llvm_modules.find(module_key);
    if (found == g_task_llvm_modules.end()) {
        return 0;
    }
    if (auto error = g_task_llvm_jit->getExecutionSession().removeJITDylib(*found->second)) {
        SetLastError(llvm::toString(std::move(error)));
        return -1;
    }
    g_task_llvm_modules.erase(found);
    return 0;
#endif
}
