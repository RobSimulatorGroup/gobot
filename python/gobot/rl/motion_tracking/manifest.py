"""Independent teacher contract: history and future motion are both mandatory."""
from dataclasses import asdict
from pathlib import Path
from gobot.rl.policy import PolicyManifest
from gobot.rl.spec import ObservationSpec, SpecField
from gobot.rl.experiment_identity import file_digest, canonical_digest
from gobot.rl.locomotion.contract import robot_physics_digest


def build_tracking_manifest(env, algorithm):
    args = asdict(algorithm) if hasattr(algorithm,'__dataclass_fields__') else dict(algorithm)
    profile = env.profile
    data = {'split':env.cfg.split,'files':{Path(p).name:file_digest(p) for p in env.cfg.motion_files},
            'source_fps':env.cfg.source_fps,'resampling_dt':env.step_dt}
    obs = ObservationSpec('g1_tracking_teacher_input_v1',(
        SpecField('proprio_history',env.observation_spec.dim*args['hist']),
        SpecField('future_goal',env.supermimic_mdp.goal_frame_dim*args['goal_frames'])))
    return PolicyManifest(task_name='g1_motion_tracking',task_version=env.cfg.task_version,
        observation_spec=obs.metadata(), action_spec=env.action_spec.metadata(), joint_names=env.joint_names,
        physics_dt=env.cfg.physics_dt,decimation=env.cfg.decimation,
        control={'mode':'position_offset','default_joint_position':profile['default_joint_position'],
                 'action_scale':profile['action_scale'],'kp':profile['kp'],'kd':profile['kd'],'action_clip':None},
        model={'type':'supermimic','history':args['hist'],'goal_frames':args['goal_frames'],
               'z_dim':args['z_dim'],'hidden_dims':args.get('test_hidden_dims') or args['policy_hidden'],
               'amp':args.get('amp','none')},
        scene_path=env.cfg.scene_path,scene_digest=env.batch.artifact.digest,
        extras={'skill':'motion_tracking','robot':env.cfg.robot_name,'data':data,'data_signature':canonical_digest(data),
                'kinematics_signature':env.supermimic_mdp.kin.digest,
                'robot_signature':robot_physics_digest(env.batch.artifact,env.cfg.robot_name),
                'actor_frame_spec':env.observation_spec.metadata(),
                'critic_spec':env.critic_observation_spec.metadata(),'solver_settings':profile['solver_settings'],
                'physics_profile':'nominal_reference_eval' if env.cfg.eval_mode else
                    'supermimic_reference_randomized_v1' if env.cfg.randomization.enabled else 'fixed_physics_smoke',
                'randomization':asdict(env.cfg.randomization),
                'ground_sliding_friction':.3 if env.cfg.randomization.enabled else 1.,
                'deterministic':env.cfg.deterministic,'runtime_quaternion':'xyzw','teacher_quaternion':'wxyz'})
