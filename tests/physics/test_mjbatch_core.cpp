// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <batch.h>

namespace {
using Model = std::unique_ptr<mjModel, mjbatch::Batch::Worker::ModelDelete>;
using Data = std::unique_ptr<mjData, mjbatch::Batch::Worker::DataDelete>;

Model MakeModel() {
    const char* xml = R"(
<mujoco><option timestep="0.002" integrator="implicitfast"/>
  <worldbody><geom type="plane" size="2 2 .1"/>
    <body pos="0 0 .25"><freejoint/><geom type="sphere" size=".1" mass="2"/>
      <site name="imu"/>
      <body pos=".2 0 0"><joint name="hinge" type="hinge" damping=".1"/>
        <geom type="capsule" size=".05 .1" mass=".5"/>
      </body>
    </body>
  </worldbody><actuator><motor joint="hinge"/></actuator>
  <sensor><gyro site="imu"/><accelerometer site="imu"/></sensor>
</mujoco>)";
    char error[1024]{};
    mjSpec* spec = mj_parseXMLString(xml, nullptr, error, sizeof(error));
    if (!spec) throw std::runtime_error(error);
    Model model(mj_compile(spec, nullptr));
    const std::string message = model ? "" : mjs_getError(spec);
    mj_deleteSpec(spec);
    if (!model) throw std::runtime_error(message);
    return model;
}

TEST(MjbatchCore, ReusedWorkersMatchIndependentMuJoCoWithPerSimulationModels) {
    auto source = MakeModel();
    constexpr int count = 9;
    mjbatch::Batch batch(source.get(), count);
    std::vector<Model> models;
    std::vector<Data> data;
    std::vector<int> ids;
    for (int i = 0; i < count; ++i) {
        models.emplace_back(mj_copyModel(nullptr, source.get()));
        data.emplace_back(mj_makeData(models.back().get()));
        auto lease = batch.Access(i);
        auto* m = batch.Model(i);
        auto* d = batch.Data(i);
        m->body_mass[1] = models[i]->body_mass[1] = 1.0 + i;
        m->body_ipos[3] = models[i]->body_ipos[3] = .002 * i;
        m->geom_friction[3] = models[i]->geom_friction[3] = .1 + .05 * i;
        mj_setConst(models[i].get(), data[i].get());
        mj_resetData(models[i].get(), data[i].get());
        d->qpos[2] = data[i]->qpos[2] = .15 + .01 * i;
        d->qvel[0] = data[i]->qvel[0] = .02 * i;
        d->ctrl[0] = data[i]->ctrl[0] = .2 * i;
        ids.push_back(i);
    }
    batch.SetConst(ids);
    batch.Run(count, 2, [&](int i) { mjbatch::Batch::Forward(batch.Model(i), batch.Data(i)); });
    for (int i = 0; i < count; ++i) mj_forward(models[i].get(), data[i].get());
    for (int iteration = 0; iteration < 60; ++iteration) {
        const int threads = iteration < 20 ? 2 : iteration < 40 ? 4 : 1;
        batch.Run(count, threads, [&](int i) {
            auto* m = batch.Model(i);
            auto* d = batch.Data(i);
            mjbatch::Batch::Step(m, d);
        });
        EXPECT_EQ(batch.num_threads(), threads);
        for (int i = 0; i < count; ++i) {
            mj_step(models[i].get(), data[i].get());
            auto lease = batch.Access(i);
            auto* d = batch.Data(i);
            for (int j = 0; j < source->nq; ++j) EXPECT_NEAR(d->qpos[j], data[i]->qpos[j], 1e-12);
            for (int j = 0; j < source->nv; ++j) EXPECT_NEAR(d->qvel[j], data[i]->qvel[j], 1e-12);
            for (int j = 0; j < source->nsensordata; ++j) EXPECT_NEAR(d->sensordata[j], data[i]->sensordata[j], 1e-12);
            for (int j = 0; j < source->nbody * 9; ++j) EXPECT_NEAR(d->xmat[j], data[i]->xmat[j], 1e-12);
            EXPECT_EQ(batch.Contacts(i).size(), data[i]->ncon);
            for (int c = 0; c < data[i]->ncon; ++c) {
                mjtNum force[6]{};
                mj_contactForce(models[i].get(), data[i].get(), c, force);
                for (int j = 0; j < 6; ++j) EXPECT_NEAR(batch.Contacts(i)[c].force[j], force[j], 1e-10);
            }
        }
    }
}

TEST(MjbatchCore, CheckpointRestoresParametersAndExactReplayAfterWorkerChange) {
    auto model = MakeModel();
    mjbatch::Batch batch(model.get(), 5);
    batch.Run(5, 2, [&](int i) {
        mjbatch::Batch::Forward(batch.Model(i), batch.Data(i));
        batch.Data(i)->ctrl[0] = .3 * i;
        for (int k = 0; k < 20; ++k) mjbatch::Batch::Step(batch.Model(i), batch.Data(i));
    });
    const auto snapshot = batch.Capture(3);
    batch.Run(5, 2, [&](int i) { for (int k = 0; k < 10; ++k) mjbatch::Batch::Step(batch.Model(i), batch.Data(i)); });
    std::vector<mjtNum> expected;
    {
        auto lease = batch.Access(3);
        expected.assign(batch.Data(3)->qpos, batch.Data(3)->qpos + model->nq);
        batch.Model(3)->body_mass[1] = 20;
    }
    batch.SetConst({3});
    const auto other = batch.Capture(2);
    batch.RestoreSnapshot(3, snapshot);
    EXPECT_EQ(batch.Capture(2), other);
    batch.Run(5, 1, [&](int i) {
        if (i == 3) for (int k = 0; k < 10; ++k) mjbatch::Batch::Step(batch.Model(i), batch.Data(i));
    });
    auto lease = batch.Access(3);
    EXPECT_DOUBLE_EQ(batch.Model(3)->body_mass[1], model->body_mass[1]);
    for (int j = 0; j < model->nq; ++j) EXPECT_DOUBLE_EQ(batch.Data(3)->qpos[j], expected[j]);
}

TEST(MjbatchCore, WorkerErrorsDoNotDeadlockAndOtherWorldsRemainIndependent) {
    auto model = MakeModel();
    mjbatch::Batch batch(model.get(), 7);
    mjbatch::Batch other(model.get(), 2);
    EXPECT_THROW(batch.Run(7, 3, [&](int i) {
        if (i == 2) throw std::runtime_error("callback failure");
        mjbatch::Batch::Step(batch.Model(i), batch.Data(i));
    }), std::runtime_error);
    EXPECT_THROW(batch.Run(7, 3, [&](int i) {
        if (i == 4) mjbatch::CheckedCall([](mjModel*, mjData*) { mju_error("native failure"); }, batch.Model(i), batch.Data(i));
    }), std::runtime_error);
    EXPECT_NO_THROW(batch.Run(7, 2, [&](int i) {
        mjbatch::Batch::Reset(batch.Model(i), batch.Data(i));
        mjbatch::Batch::Step(batch.Model(i), batch.Data(i));
    }));
    auto access = other.Access(0);
    EXPECT_DOUBLE_EQ(other.Data(0)->time, 0);
    EXPECT_THROW(other.Model(1), std::logic_error);
}

TEST(MjbatchCore, SleepAndInvalidDimensionsAreRejected) {
    auto model = MakeModel();
    EXPECT_THROW(mjbatch::Batch(model.get(), 0), std::invalid_argument);
    model->opt.enableflags |= mjENBL_SLEEP;
    EXPECT_THROW(mjbatch::Batch(model.get(), 2), std::invalid_argument);
}

TEST(MjbatchCore, PartialResetPreservesOtherEnvironmentsAndModelOverrides) {
    auto model = MakeModel();
    mjbatch::Batch batch(model.get(), 7);
    batch.Run(7, 2, [&](int i) {
        batch.Model(i)->geom_friction[3] = .2 + .1 * i;
        batch.Data(i)->ctrl[0] = .1 * i;
        for (int k = 0; k < 10; ++k) mjbatch::Batch::Step(batch.Model(i), batch.Data(i));
    });
    const auto before = batch.Capture(4);
    {
        auto lease = batch.Access(3);
        mjbatch::Batch::Reset(batch.Model(3), batch.Data(3));
        mjbatch::Batch::Forward(batch.Model(3), batch.Data(3));
        EXPECT_DOUBLE_EQ(batch.Data(3)->time, 0);
        EXPECT_DOUBLE_EQ(batch.Data(3)->ctrl[0], 0);
        EXPECT_DOUBLE_EQ(batch.Model(3)->geom_friction[3], .5);
    }
    EXPECT_EQ(batch.Capture(4), before);
}
} // namespace
