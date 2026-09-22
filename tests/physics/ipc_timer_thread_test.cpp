// Explicit SDK contract test; does not need a CUDA device.
#include <uipc/common/timer.h>
#include <barrier>
#include <future>
#include <stdexcept>
#include <iostream>

int main() {
    try {
        std::barrier rendezvous(2);
        auto* owner_timer = uipc::GlobalTimer::current();
        auto run = [&](const char* name) {
            if (uipc::Timer::enabled()) throw std::runtime_error("profiling leaked into worker");
            uipc::GlobalTimer timer(name);
            timer.set_as_current();
            int syncs = 0;
            uipc::Timer::set_sync_func([&] { ++syncs; });
            { uipc::Timer disabled("disabled"); }
            if (syncs) throw std::runtime_error("disabled timer synchronized");
            uipc::Timer::enable_all();
            try {
                uipc::Timer scope(name);
                rendezvous.arrive_and_wait(); // Both stacks must be active.
                throw std::runtime_error("injected solve error");
            } catch (const std::runtime_error&) {}
            const auto report = timer.report_merged_as_json();
            if (syncs != 2 || report["children"].size() != 1 ||
                report["children"][0]["name"] != name)
                throw std::runtime_error("timer scopes crossed threads or failed to unwind");
            uipc::Timer::disable_all();
            uipc::Timer::set_sync_func({});
        };
        auto a = std::async(std::launch::async, run, "world A");
        auto b = std::async(std::launch::async, run, "world B");
        a.get();
        b.get();
        if (uipc::Timer::enabled() || uipc::GlobalTimer::current() != owner_timer)
            throw std::runtime_error("profiling changed owner thread state");
        std::cout << "IPC timer thread isolation and exception cleanup passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
