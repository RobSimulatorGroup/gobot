// Translation unit for the standalone C++ gobot_planning library (zero Python dependency).
// Instantiates key entry points so C++ callers (gobot_editor, C++ tests, _coal_native)
// link against a single compiled library target.

#include "planning_planner.hpp"
#include "planning_shapes.hpp"

namespace gobot::planning {

int gobot_planning_abi_version() noexcept {
    return 1;
}

bool gobot_planning_has_tbb() noexcept {
#ifdef GOBOT_PLANNING_HAS_TBB
    return true;
#else
    return false;
#endif
}

}  // namespace gobot::planning
