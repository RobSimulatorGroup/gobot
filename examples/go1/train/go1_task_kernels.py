"""Task kernels for the Go1 velocity example."""

from __future__ import annotations

from gobot.rl import dim, flag, kernel, param, tid, weight, where


R_TRACK_LINEAR_VELOCITY = 0
R_TRACK_ANGULAR_VELOCITY = 1
R_UPRIGHT = 2
R_POSE = 3
R_BODY_ANG_VEL = 4
R_DOF_POS_LIMITS = 5
R_ACTION_RATE_L2 = 6
R_AIR_TIME = 7
R_FOOT_CLEARANCE = 8
R_FOOT_SWING_HEIGHT = 9
R_FOOT_SLIP = 10
R_SOFT_LANDING = 11
R_SELF_COLLISIONS = 12
R_SHANK_COLLISION = 13
R_TRUNK_HEAD_COLLISION = 14

P_STEP_DT = 0
P_LIN_VEL_STD2 = 1
P_ANG_VEL_STD2 = 2
P_UPRIGHT_STD2 = 3
P_FOOT_TARGET_HEIGHT = 4
P_COMMAND_THRESHOLD = 5
P_MIN_BASE_CLEARANCE = 6
P_FLAT_ROLL_PITCH_LIMIT = 7
P_POSE_WALKING_THRESHOLD = 8
P_POSE_RUNNING_THRESHOLD = 9
P_HEIGHT_SCAN_MAX_DISTANCE = 10

F_ROUGH_TERRAIN = 0


@kernel
def go1_velocity_task(a):
    env_id = tid()
    env3 = env_id * 3
    num_dof = dim(a.joint_position, 1)
    num_feet = dim(a.foot_height, 1)
    height_scan_dim = dim(a.height_scan, 1)
    actor_obs_dim = dim(a.actor_obs, 1)
    critic_obs_dim = dim(a.critic_obs, 1)
    reward_term_dim = dim(a.reward_terms, 1)

    step_dt = param(a.task_params, P_STEP_DT, 0.02)
    lin_vel_std2 = max(param(a.task_params, P_LIN_VEL_STD2, 0.25), 0.000001)
    ang_vel_std2 = max(param(a.task_params, P_ANG_VEL_STD2, 0.5), 0.000001)
    upright_std2 = max(param(a.task_params, P_UPRIGHT_STD2, 0.2), 0.000001)
    command_threshold = param(a.task_params, P_COMMAND_THRESHOLD, 0.05)
    min_base_clearance = param(a.task_params, P_MIN_BASE_CLEARANCE, 0.16)
    flat_limit = param(a.task_params, P_FLAT_ROLL_PITCH_LIMIT, 1.2217305)
    height_scan_scale = 1.0 / max(param(a.task_params, P_HEIGHT_SCAN_MAX_DISTANCE, 5.0), 0.000001)

    command0 = a.command[env3 + 0]
    command1 = a.command[env3 + 1]
    command2 = a.command[env3 + 2]
    lin0 = a.base_linear_velocity_body[env3 + 0]
    lin1 = a.base_linear_velocity_body[env3 + 1]
    lin2 = a.base_linear_velocity_body[env3 + 2]
    ang0 = a.base_angular_velocity_body[env3 + 0]
    ang1 = a.base_angular_velocity_body[env3 + 1]
    ang2 = a.base_angular_velocity_body[env3 + 2]

    lin_error = (command0 - lin0) * (command0 - lin0) + (command1 - lin1) * (command1 - lin1) + lin2 * lin2
    ang_error = (command2 - ang2) * (command2 - ang2) + ang0 * ang0 + ang1 * ang1
    command_speed = sqrt(command0 * command0 + command1 * command1) + abs(command2)
    active = where(command_speed > command_threshold, 1.0, 0.0)
    upright_error = (
        a.projected_gravity[env3 + 0] * a.projected_gravity[env3 + 0]
        + a.projected_gravity[env3 + 1] * a.projected_gravity[env3 + 1]
    )

    action_rate = 0.0
    for joint_index in range(num_dof):
        dof_offset = env_id * num_dof + joint_index
        delta_action = a.submitted_action[dof_offset] - a.previous_action[dof_offset]
        action_rate += delta_action * delta_action

    foot_clearance_sum = 0.0
    foot_slip_sum = 0.0
    air_time_count = 0.0
    for foot_index in range(num_feet):
        foot = env_id * num_feet + foot_index
        if a.foot_air_time[foot] > 0.05 and a.foot_air_time[foot] < 0.5:
            air_time_count += 1.0
        foot_clearance_sum += a.foot_height[foot]
        foot_slip_sum += where(a.foot_contact[foot] > 0.0, 1.0, 0.0)

    a.foot_slip[env_id] = foot_slip_sum * active
    a.base_clearance[env_id] = a.base_height[env_id]
    a.velocity_error[env_id] = sqrt((command0 - lin0) * (command0 - lin0) + (command1 - lin1) * (command1 - lin1))

    term_base = env_id * reward_term_dim
    for term_index in range(reward_term_dim):
        a.reward_terms[term_base + term_index] = 0.0
    a.reward_terms[term_base + R_TRACK_LINEAR_VELOCITY] = weight(a.reward_weights, R_TRACK_LINEAR_VELOCITY) * exp(-lin_error / lin_vel_std2)
    a.reward_terms[term_base + R_TRACK_ANGULAR_VELOCITY] = weight(a.reward_weights, R_TRACK_ANGULAR_VELOCITY) * exp(-ang_error / ang_vel_std2)
    a.reward_terms[term_base + R_UPRIGHT] = weight(a.reward_weights, R_UPRIGHT) * exp(-upright_error / upright_std2)
    a.reward_terms[term_base + R_ACTION_RATE_L2] = weight(a.reward_weights, R_ACTION_RATE_L2) * action_rate
    a.reward_terms[term_base + R_AIR_TIME] = weight(a.reward_weights, R_AIR_TIME) * air_time_count * active
    a.reward_terms[term_base + R_FOOT_CLEARANCE] = weight(a.reward_weights, R_FOOT_CLEARANCE) * foot_clearance_sum * active
    a.reward_terms[term_base + R_FOOT_SLIP] = weight(a.reward_weights, R_FOOT_SLIP) * foot_slip_sum * active

    reward_sum = 0.0
    for term_index in range(reward_term_dim):
        reward_sum += a.reward_terms[term_base + term_index]
    a.reward[env_id] = reward_sum * step_dt

    has_terminated = a.base_clearance[env_id] < min_base_clearance
    if flag(a.task_flags, F_ROUGH_TERRAIN) <= 0.0:
        has_terminated = has_terminated or abs(a.projected_gravity[env3 + 0]) > sin(flat_limit)
        has_terminated = has_terminated or abs(a.projected_gravity[env3 + 1]) > sin(flat_limit)
    a.terminated[env_id] = where(has_terminated, 1, 0)

    write = env_id * actor_obs_dim
    for axis in range(3):
        a.actor_obs[write] = a.base_linear_velocity_body[env3 + axis]
        write += 1
    for axis in range(3):
        a.actor_obs[write] = a.base_angular_velocity_body[env3 + axis]
        write += 1
    for axis in range(3):
        a.actor_obs[write] = a.projected_gravity[env3 + axis]
        write += 1
    for joint_index in range(num_dof):
        dof_offset = env_id * num_dof + joint_index
        a.actor_obs[write] = a.joint_position[dof_offset] + a.encoder_bias[dof_offset] - a.default_joint_position[joint_index]
        write += 1
    for joint_index in range(num_dof):
        a.actor_obs[write] = a.joint_velocity[env_id * num_dof + joint_index]
        write += 1
    for joint_index in range(num_dof):
        a.actor_obs[write] = a.last_action[env_id * num_dof + joint_index]
        write += 1
    for axis in range(3):
        a.actor_obs[write] = a.command[env3 + axis]
        write += 1
    for sample_index in range(height_scan_dim):
        a.actor_obs[write] = a.height_scan[env_id * height_scan_dim + sample_index] * height_scan_scale
        write += 1

    critic_write = env_id * critic_obs_dim
    actor_start = env_id * actor_obs_dim
    for obs_index in range(actor_obs_dim):
        a.critic_obs[critic_write] = a.actor_obs[actor_start + obs_index]
        critic_write += 1
    for foot_index in range(num_feet):
        a.critic_obs[critic_write] = a.foot_height[env_id * num_feet + foot_index]
        critic_write += 1
    for foot_index in range(num_feet):
        a.critic_obs[critic_write] = a.foot_air_time[env_id * num_feet + foot_index]
        critic_write += 1
    for foot_index in range(num_feet):
        a.critic_obs[critic_write] = a.foot_contact[env_id * num_feet + foot_index]
        critic_write += 1
    for foot_index in range(num_feet):
        foot3 = (env_id * num_feet + foot_index) * 3
        for axis in range(3):
            contact_force = a.foot_contact_force[foot3 + axis]
            a.critic_obs[critic_write] = copysign(log1p(abs(contact_force)), contact_force)
            critic_write += 1
