/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/main/engine_context.hpp"
#include "gobot/scene/scene_command.hpp"

#include <memory>
#include <string>
#include <utility>

namespace gobot {

// Manages a single interactive drag gesture backed by SceneCommandStack transactions:
// - Begins a transaction lazily on the first actual scene mutation.
// - Cancels and rolls back immediately on Esc, latching gesture_cancelled_ so subsequent
//   frames while the mouse button remains held cannot re-apply the cancelled drag.
// - Commits cleanly on gesture release (or clears the cancel latch once released).
class EditorDragTransactionController {
public:
    [[nodiscard]] bool IsTransactionActive() const noexcept { return transaction_active_; }
    [[nodiscard]] bool IsGestureCancelled() const noexcept { return gesture_cancelled_; }

    bool ExecuteMutation(EngineContext* context,
                         const std::string& transaction_name,
                         std::unique_ptr<SceneCommand> command) {
        if (context == nullptr || !command || gesture_cancelled_) {
            return false;
        }
        if (!transaction_active_) {
            if (!context->BeginSceneTransaction(transaction_name)) {
                return false;
            }
            transaction_active_ = true;
        }
        return context->ExecuteSceneCommand(std::move(command));
    }

    bool ExecuteMutation(SceneCommandStack* stack,
                         const std::string& transaction_name,
                         std::unique_ptr<SceneCommand> command) {
        if (stack == nullptr || !command || gesture_cancelled_) {
            return false;
        }
        if (!transaction_active_) {
            if (!stack->BeginTransaction(transaction_name)) {
                return false;
            }
            transaction_active_ = true;
        }
        return stack->Execute(std::move(command));
    }

    bool CancelOnEscape(EngineContext* context) {
        if (!transaction_active_) {
            return false;
        }
        const bool ok = (context != nullptr) ? context->CancelSceneTransaction() : false;
        transaction_active_ = false;
        gesture_cancelled_ = true;
        return ok;
    }

    bool CancelOnEscape(SceneCommandStack* stack) {
        if (!transaction_active_) {
            return false;
        }
        const bool ok = (stack != nullptr) ? stack->CancelTransaction() : false;
        transaction_active_ = false;
        gesture_cancelled_ = true;
        return ok;
    }

    bool CommitAndReset(EngineContext* context) {
        bool ok = true;
        if (transaction_active_) {
            ok = (context != nullptr) ? context->CommitSceneTransaction() : false;
            transaction_active_ = false;
        }
        gesture_cancelled_ = false;
        return ok;
    }

    bool CommitAndReset(SceneCommandStack* stack) {
        bool ok = true;
        if (transaction_active_) {
            ok = (stack != nullptr) ? stack->CommitTransaction() : false;
            transaction_active_ = false;
        }
        gesture_cancelled_ = false;
        return ok;
    }

private:
    bool transaction_active_{false};
    bool gesture_cancelled_{false};
};

} // namespace gobot
