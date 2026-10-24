// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <mujoco/mjbatch.h>
#include <mujoco/mujoco.h>
#include <mujoco/mjxmacro.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace gobot::mujoco_detail {

// Engine-private adapter. MuJoCo owns the pool, workspaces, integration rows and
// expanded parameters. Only the derived fields Gobot reads and variable-length
// contact wrenches are retained between visits to a reused workspace.
class MuJoCoBatch {
public:
    struct ModelDelete { void operator()(mjModel* p) const { mj_deleteModel(p); } };
    struct DataDelete { void operator()(mjData* p) const { mj_deleteData(p); } };
    struct BatchDelete { void operator()(mjBatch* p) const { mjb_deleteBatch(p); } };
    struct Contact { mjContact value{}; std::array<mjtNum, 6> force{}; };
    using Task = std::function<void(int)>;

private:
    struct Context {
        MuJoCoBatch* owner;
        int sim;
        mjModel* model;
        mjData* data;
        bool callback;
    };
    inline static thread_local Context* current_ = nullptr;
    struct Output {
        mjtNum* mjData::* member;
        mjtNum* rows;
        std::size_t size;
    };
    struct Parameter {
        mjtNum* mjModel::* member;
        mjtNum* rows;
        std::size_t size;
    };
    struct Option {
        std::size_t offset;
        std::size_t bytes;
        std::uint8_t* rows;
    };

public:
    // Individual engine queries use one scratch model/data, with nested visits
    // reusing the current view. Native operations catch errors before returning
    // through this C++ scope; no lease is alive inside the raw C trampoline.
    class Lease {
    public:
        Lease(MuJoCoBatch* batch, int sim) : batch_(batch), previous_(current_) {
            if (!batch_) return;
            batch_->CheckIndex(sim);
            if (current_ && current_->owner == batch_) {
                if (current_->sim != sim) throw std::logic_error("nested MuJoCo visit changed environment");
                return;
            }
            lock_ = std::unique_lock<std::mutex>(batch_->mutex_);
            batch_->LoadScratch(sim);
            context_ = {batch_, sim, batch_->model_.get(), batch_->data_.get(), false};
            current_ = &context_;
            owns_ = true;
        }
        ~Lease() {
            if (owns_) {
                batch_->SaveScratch(context_.sim);
                current_ = previous_;
            }
        }
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
    private:
        MuJoCoBatch* batch_;
        Context* previous_;
        Context context_{};
        bool owns_{false};
        std::unique_lock<std::mutex> lock_;
    };

    explicit MuJoCoBatch(const mjModel* model, int count) {
        if (!model || count <= 0) throw std::invalid_argument("MuJoCo batch size must be positive");
        if (model->opt.enableflags & (mjENBL_SLEEP | mjENBL_IPC))
            throw std::invalid_argument("MuJoCo CPU batches do not support sleep or native IPC");
        if (mj_version() != mjVERSION_HEADER) throw std::runtime_error("MuJoCo header/runtime version mismatch");
        model_.reset(mj_copyModel(nullptr, model));
        data_.reset(mj_makeData(model));
        if (!model_ || !data_) throw std::bad_alloc();
        batch_ = MakeBatch(model, count, 1);
        contacts_.resize(count);
        errors_.resize(count);
        BindFields();
    }
    MuJoCoBatch(const MuJoCoBatch&) = delete;
    MuJoCoBatch& operator=(const MuJoCoBatch&) = delete;

    int num_sims() const { return mjb_nsim(batch_.get()); }
    int num_threads() const { return mjb_nthread(batch_.get()); }
    Lease Access(int sim) { return Lease(this, sim); }
    mjModel* Model(int sim) const { return Active(sim).model; }
    mjData* Data(int sim) const { return Active(sim).data; }
    const std::vector<Contact>& Contacts(int sim) const { return contacts_.at(sim); }

    static void Step(mjModel* m, mjData* d) { CurrentOperation(m, d, Operation::Step); }
    static void Forward(mjModel* m, mjData* d) { CurrentOperation(m, d, Operation::Forward); }
    static void Reset(mjModel* m, mjData* d) { CurrentOperation(m, d, Operation::Reset); }

    // These callbacks do engine work only. Physics calls belong in Advance so
    // a MuJoCo longjmp never crosses a std::function or C++ automatic object.
    void Run(int count, int threads, const Task& task) {
        Dispatch(count, threads, 0, task, {}, {});
    }
    void Advance(int count, int threads, std::uint64_t ticks,
                 const Task& before = {}, const Task& after = {}, const Task& finish = {}) {
        Dispatch(count, threads, ticks, before, after, finish);
    }
    void SetConst(const std::vector<int>& ids) {
        if (ids.empty()) return;
        std::lock_guard lock(mutex_);
        CheckIds(ids);
        const int failed = mjb_setConst(batch_.get(), ids.data(), static_cast<int>(ids.size()));
        CheckNativeErrors(failed, ids);
    }

    std::vector<std::uint8_t> Capture(int sim) {
        CheckIndex(sim);
        if (current_ && current_->owner == this && !current_->callback) SaveScratch(current_->sim);
        std::vector<std::uint8_t> bytes;
        const auto put = [&](const void* p, std::size_t size) {
            const auto* begin = static_cast<const std::uint8_t*>(p);
            if (size) bytes.insert(bytes.end(), begin, begin + size);
        };
        const std::uint32_t header[] = {2, mjVERSION_HEADER, static_cast<std::uint32_t>(mjb_nstate(batch_.get()))};
        put(header, sizeof(header));
        put(State(sim), StateBytes());
        put(mjb_warning(batch_.get()) + sim * mjNWARNING, sizeof(mjWarningStat) * mjNWARNING);
        for (const auto& p : parameters_) put(p.rows + sim * p.size, p.size * sizeof(mjtNum));
        for (const auto& o : options_) put(o.rows + sim * o.bytes, o.bytes);
        for (const auto& o : outputs_) put(o.rows + sim * o.size, o.size * sizeof(mjtNum));
        const std::uint64_t count = contacts_[sim].size();
        put(&count, sizeof(count));
        put(contacts_[sim].data(), count * sizeof(Contact));
        return bytes;
    }

    void RestoreSnapshot(int sim, std::span<const std::uint8_t> bytes) {
        std::lock_guard lock(mutex_);
        Restore(sim, bytes);
    }

private:
    enum class Operation { Step, Forward, Reset };
    struct Call {
        MuJoCoBatch* owner;
        std::uint64_t ticks;
        const Task* before;
        const Task* after;
        const Task* finish;
        Operation operation;
        bool raw;
    };

    static std::unique_ptr<mjBatch, BatchDelete> MakeBatch(const mjModel* model, int count, int threads) {
        mjBatchConfig config;
        mjb_defaultConfig(&config);
        config.nthread = threads;
        config.persistent = false;
        char error[1024]{};
        std::unique_ptr<mjBatch, BatchDelete> b(mjb_makeBatch(model, count, &config, error, sizeof(error)));
        if (!b) throw std::runtime_error(error);
        return b;
    }
    void BindFields() {
        parameters_.clear(); options_.clear(); outputs_.clear();
        const auto parameter = [&](const char* name, mjtNum* mjModel::* member) {
            int size = 0, element = 0;
            auto* rows = static_cast<mjtNum*>(mjb_expand(batch_.get(), name, &size, &element));
            if (!rows || element != sizeof(mjtNum)) throw std::runtime_error(std::string("invalid MuJoCo parameter: ") + name);
            parameters_.push_back({member, rows, static_cast<std::size_t>(size)});
        };
        parameter("body_mass", &mjModel::body_mass);
        parameter("body_ipos", &mjModel::body_ipos);
        parameter("geom_friction", &mjModel::geom_friction);
        parameter("actuator_gainprm", &mjModel::actuator_gainprm);
        parameter("actuator_biasprm", &mjModel::actuator_biasprm);
#define X(type, name, count) \
        options_.push_back({offsetof(mjOption, name), sizeof(type) * count, \
            static_cast<std::uint8_t*>(mjb_expand(batch_.get(), "opt." #name, nullptr, nullptr))});
#define XVEC X
        MJOPTION_FIELDS
#undef XVEC
#undef X
        for (const auto& o : options_) if (!o.rows) throw std::runtime_error("invalid MuJoCo option field");
        const auto output = [&](const char* name, mjtNum* mjData::* member) {
            int size = 0, element = 0;
            auto* rows = static_cast<mjtNum*>(mjb_output(batch_.get(), name, &size, &element));
            if ((!rows && size) || element != sizeof(mjtNum)) throw std::runtime_error(std::string("invalid MuJoCo output: ") + name);
            outputs_.push_back({member, rows, static_cast<std::size_t>(size)});
        };
        output("xpos", &mjData::xpos); output("xmat", &mjData::xmat);
        output("xipos", &mjData::xipos); output("cvel", &mjData::cvel);
        output("subtree_com", &mjData::subtree_com); output("qacc", &mjData::qacc);
        output("qfrc_actuator", &mjData::qfrc_actuator); output("sensordata", &mjData::sensordata);
        output("geom_xpos", &mjData::geom_xpos); output("geom_xmat", &mjData::geom_xmat);
    }
    void CheckIndex(int sim) const {
        if (sim < 0 || sim >= num_sims()) throw std::out_of_range("MuJoCo environment index");
    }
    void CheckIds(const std::vector<int>& ids) const {
        int last = -1;
        for (int sim : ids) {
            CheckIndex(sim);
            if (sim <= last) throw std::invalid_argument("MuJoCo selection must be sorted and unique");
            last = sim;
        }
    }
    Context& Active(int sim) const {
        if (!current_ || current_->owner != this || current_->sim != sim)
            throw std::logic_error("MuJoCo workspace accessed outside a visit");
        return *current_;
    }
    mjtNum* State(int sim) { return mjb_state(batch_.get()) + sim * static_cast<std::size_t>(mjb_nstate(batch_.get())); }
    std::size_t StateBytes() const { return sizeof(mjtNum) * mjb_nstate(batch_.get()); }
    void LoadOutputs(mjData* d, int sim) const {
        for (const auto& o : outputs_) if (o.size) std::memcpy(d->*o.member, o.rows + sim * o.size, o.size * sizeof(mjtNum));
    }
    void SaveOutputs(const mjData* d, int sim) {
        for (const auto& o : outputs_) if (o.size) std::memcpy(o.rows + sim * o.size, d->*o.member, o.size * sizeof(mjtNum));
    }
    void StoreOptions(const mjModel* m, int sim) {
        for (const auto& o : options_)
            std::memcpy(o.rows + sim * o.bytes, reinterpret_cast<const std::uint8_t*>(&m->opt) + o.offset, o.bytes);
    }
    void LoadScratch(int sim) {
        // Queries need structural metadata plus the explicitly mutable fields,
        // not another copy of all meshes/heightfields for every environment.
        for (const auto& p : parameters_) if (p.size)
            std::memcpy(model_.get()->*p.member, p.rows + sim * p.size, p.size * sizeof(mjtNum));
        for (const auto& o : options_)
            std::memcpy(reinterpret_cast<std::uint8_t*>(&model_->opt) + o.offset, o.rows + sim * o.bytes, o.bytes);
        mjb_getData(batch_.get(), sim, data_.get());
        LoadOutputs(data_.get(), sim);
    }
    void SaveScratch(int sim) {
        mj_getState(model_.get(), data_.get(), State(sim), mjSTATE_INTEGRATION);
        std::memcpy(mjb_warning(batch_.get()) + sim * mjNWARNING, data_->warning, sizeof(data_->warning));
        for (const auto& p : parameters_) if (p.size)
            std::memcpy(p.rows + sim * p.size, model_.get()->*p.member, p.size * sizeof(mjtNum));
        StoreOptions(model_.get(), sim);
        SaveOutputs(data_.get(), sim);
    }
    void ModelView(mjModel* m, int sim) {
        // Shallow header: the native model is never modified. Mutable fields
        // point directly to this environment's officially expanded parameter row.
        for (const auto& p : parameters_) m->*p.member = p.rows + sim * p.size;
    }
    bool Invoke(const Task* task, Context* context) noexcept {
        if (!task || !*task) return true;
        Context* previous = current_;
        current_ = context;
        try { (*task)(context->sim); }
        catch (const std::exception& e) { errors_[context->sim] = e.what(); }
        catch (...) { errors_[context->sim] = "unknown engine callback failure"; }
        current_ = previous;
        return errors_[context->sim].empty();
    }
    bool Cache(const mjModel* m, mjData* d, int sim) noexcept {
        try {
            auto& contacts = contacts_[sim];
            contacts.resize(d->ncon);
            for (int i = 0; i < d->ncon; ++i) {
                contacts[i].value = d->contact[i];
                mj_contactForce(m, d, i, contacts[i].force.data());
            }
            SaveOutputs(d, sim);
            return true;
        } catch (...) { errors_[sim] = "failed to capture MuJoCo results"; return false; }
    }
    static bool ValidStep(const mjModel* m, const mjData* d, double time, const int* warning) {
        if (!std::isfinite(d->time) || std::abs(d->time - time - m->opt.timestep) > std::max(1e-9, m->opt.timestep * 1e-5)) return false;
        const int kinds[] = {mjWARN_BADQPOS, mjWARN_BADQVEL, mjWARN_BADQACC};
        for (int i = 0; i < 3; ++i) if (d->warning[kinds[i]].number != warning[i]) return false;
        for (int i = 0; i < m->nq; ++i) if (!std::isfinite(d->qpos[i])) return false;
        for (int i = 0; i < m->nv; ++i) if (!std::isfinite(d->qvel[i]) || !std::isfinite(d->qacc[i])) return false;
        return true;
    }
    static void Apply(const mjModel* native, mjData* data, int sim, void* arg) {
        auto* call = static_cast<Call*>(arg);
        auto* self = call->owner;
        mjModel model = *native;
        Context context{self, sim, &model, data, true};
        static_assert(std::is_trivially_destructible_v<decltype(model)>);
        static_assert(std::is_trivially_destructible_v<decltype(context)>);
        self->ModelView(&model, sim);
        self->LoadOutputs(data, sim);
        if (call->raw) {
            if (call->operation == Operation::Step) mj_step(&model, data);
            else mj_forward(&model, data);
            self->Cache(&model, data, sim);
            return;
        }
        for (std::uint64_t tick = 0; tick < std::max<std::uint64_t>(1, call->ticks); ++tick) {
            if (!self->Invoke(call->before, &context)) break;
            self->StoreOptions(&model, sim);
            if (model.opt.enableflags & (mjENBL_SLEEP | mjENBL_IPC)) {
                self->errors_[sim] = "MuJoCo CPU batches do not support sleep or native IPC";
                break;
            }
            if (call->ticks) {
                const double time = data->time;
                const int warning[] = {data->warning[mjWARN_BADQPOS].number,
                    data->warning[mjWARN_BADQVEL].number, data->warning[mjWARN_BADQACC].number};
                // No C++ object requiring destruction is live between this call
                // and mjb_apply's own error trap. Engine callbacks have returned.
                mj_step(&model, data);
                if (!ValidStep(&model, data, time, warning)) {
                    self->errors_[sim] = "MuJoCo step reset or produced invalid state; reset is required.";
                    break;
                }
                if (!self->Cache(&model, data, sim)) break;
            }
            if (!self->Invoke(call->after, &context)) break;
        }
        if (self->errors_[sim].empty()) self->Invoke(call->finish, &context);
        if (!self->errors_[sim].empty()) mj_setState(&model, data, self->State(sim), mjSTATE_INTEGRATION);
        else self->SaveOutputs(data, sim);
    }
    void CheckNativeErrors(int failed, std::span<const int> ids) const {
        if (!failed) return;
        for (int sim : ids) if (mjb_status(batch_.get())[sim])
            throw std::runtime_error("environment " + std::to_string(sim) + ": " + mjb_error(batch_.get(), sim));
    }
    void CheckCallbackErrors() const {
        for (int sim = 0; sim < num_sims(); ++sim) if (!errors_[sim].empty())
            throw std::runtime_error("environment " + std::to_string(sim) + ": " + errors_[sim]);
    }
    static void CurrentOperation(mjModel* model, mjData* data, Operation operation) {
        if (!current_ || current_->model != model || current_->data != data || current_->callback)
            throw std::logic_error("MuJoCo physics must run outside engine callbacks");
        auto* self = current_->owner;
        const int sim = current_->sim;
        self->SaveScratch(sim);
        self->errors_[sim].clear();
        int failed;
        if (operation == Operation::Reset) {
            failed = mjb_reset(self->batch_.get(), &sim, 1, nullptr, mjSTAGE_NONE);
            self->contacts_[sim].clear();
        } else {
            Call call{self, 0, nullptr, nullptr, nullptr, operation, true};
            failed = mjb_apply(self->batch_.get(), &sim, 1, Apply, &call, 1);
        }
        self->LoadScratch(sim);
        self->CheckNativeErrors(failed, {&sim, 1});
        if (!self->errors_[sim].empty())
            throw std::runtime_error("environment " + std::to_string(sim) + ": " + self->errors_[sim]);
    }
    void Dispatch(int count, int threads, std::uint64_t ticks, const Task& before,
                  const Task& after, const Task& finish) {
        std::lock_guard lock(mutex_);
        if (count < 0 || count > num_sims()) throw std::out_of_range("MuJoCo batch count");
        if (!count) return;
        ConfigureWorkers(std::clamp(threads, 1, count));
        for (auto& error : errors_) error.clear();
        std::vector<int> ids(count);
        for (int i = 0; i < count; ++i) ids[i] = i;
        Call call{this, ticks, &before, &after, &finish, Operation::Step, false};
        const int failed = mjb_apply(batch_.get(), ids.data(), count, Apply, &call, 1);
        CheckNativeErrors(failed, ids);
        CheckCallbackErrors();
    }
    void ConfigureWorkers(int threads) {
        if (threads == num_threads()) return;
        std::vector<std::vector<std::uint8_t>> saved;
        for (int i = 0; i < num_sims(); ++i) saved.push_back(Capture(i));
        // Allocate first; allocation failure leaves the active batch intact.
        auto next = MakeBatch(mjb_model(batch_.get()), num_sims(), threads);
        batch_ = std::move(next);
        BindFields();
        for (int i = 0; i < num_sims(); ++i) Restore(i, saved[i]);
    }
    void Restore(int sim, std::span<const std::uint8_t> bytes) {
        CheckIndex(sim);
        const auto require = [&](std::size_t size) {
            if (bytes.size() < size) throw std::invalid_argument("truncated MuJoCo checkpoint");
        };
        const std::uint32_t header[] = {2, mjVERSION_HEADER, static_cast<std::uint32_t>(mjb_nstate(batch_.get()))};
        require(sizeof(header));
        if (std::memcmp(bytes.data(), header, sizeof(header))) throw std::invalid_argument("incompatible MuJoCo checkpoint");
        std::size_t fixed = sizeof(header) + StateBytes() + sizeof(mjWarningStat) * mjNWARNING;
        for (const auto& p : parameters_) fixed += p.size * sizeof(mjtNum);
        for (const auto& o : options_) fixed += o.bytes;
        for (const auto& o : outputs_) fixed += o.size * sizeof(mjtNum);
        require(fixed + sizeof(std::uint64_t));
        std::uint64_t count;
        std::memcpy(&count, bytes.data() + fixed, sizeof(count));
        if (count != (bytes.size() - fixed - sizeof(count)) / sizeof(Contact) ||
            (bytes.size() - fixed - sizeof(count)) % sizeof(Contact))
            throw std::invalid_argument("invalid MuJoCo checkpoint contacts");
        std::vector<Contact> contacts(count);
        std::size_t offset = sizeof(header) + StateBytes() + sizeof(mjWarningStat) * mjNWARNING;
        const auto get = [&](void* p, std::size_t size) {
            if (size) std::memcpy(p, bytes.data() + offset, size);
            offset += size;
        };
        for (const auto& p : parameters_) get(p.rows + sim * p.size, p.size * sizeof(mjtNum));
        for (const auto& o : options_) get(o.rows + sim * o.bytes, o.bytes);
        CheckNativeErrors(mjb_setConst(batch_.get(), &sim, 1), {&sim, 1});
        for (const auto& o : outputs_) get(o.rows + sim * o.size, o.size * sizeof(mjtNum));
        offset += sizeof(count);
        get(contacts.data(), count * sizeof(Contact));
        contacts_[sim] = std::move(contacts);
        std::memcpy(State(sim), bytes.data() + sizeof(header), StateBytes());
        std::memcpy(mjb_warning(batch_.get()) + sim * mjNWARNING,
                    bytes.data() + sizeof(header) + StateBytes(), sizeof(mjWarningStat) * mjNWARNING);
        errors_[sim].clear();
    }

    std::unique_ptr<mjBatch, BatchDelete> batch_;
    std::unique_ptr<mjModel, ModelDelete> model_;
    std::unique_ptr<mjData, DataDelete> data_;
    std::vector<Parameter> parameters_;
    std::vector<Option> options_;
    std::vector<Output> outputs_;
    std::vector<std::vector<Contact>> contacts_;
    std::vector<std::string> errors_;
    std::mutex mutex_;
};
} // namespace gobot::mujoco_detail
