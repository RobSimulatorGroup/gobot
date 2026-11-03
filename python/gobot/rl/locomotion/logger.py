"""Keep learner log aggregation off CUDA."""
import torch

def _install_cpu_logger_bookkeeping(logger) -> None:
    logger.device = "cpu"
    logger.cur_reward_sum = logger.cur_reward_sum.cpu()
    logger.cur_episode_length = logger.cur_episode_length.cpu()
    if hasattr(logger, "cur_ereward_sum"):
        logger.cur_ereward_sum = logger.cur_ereward_sum.cpu()
        logger.cur_ireward_sum = logger.cur_ireward_sum.cpu()

    def process_env_step_cpu(rewards, dones, extras, intrinsic_rewards=None) -> None:
        if logger.writer is None:
            return
        if "episode" in extras:
            logger.ep_extras.append(_cpu_log_mapping(extras["episode"]))
        elif "log" in extras:
            logger.ep_extras.append(_cpu_log_mapping(extras["log"]))

        rewards_cpu = _cpu_tensor(rewards, dtype=torch.float32)
        dones_cpu = _cpu_tensor(dones, dtype=torch.bool)
        if intrinsic_rewards is not None:
            intrinsic_rewards_cpu = _cpu_tensor(intrinsic_rewards, dtype=torch.float32)
            logger.cur_ereward_sum += rewards_cpu
            logger.cur_ireward_sum += intrinsic_rewards_cpu
            logger.cur_reward_sum += rewards_cpu + intrinsic_rewards_cpu
        else:
            logger.cur_reward_sum += rewards_cpu
        logger.cur_episode_length += 1

        new_ids = (dones_cpu > 0).nonzero(as_tuple=False)
        logger.rewbuffer.extend(logger.cur_reward_sum[new_ids][:, 0].numpy().tolist())
        logger.lenbuffer.extend(logger.cur_episode_length[new_ids][:, 0].numpy().tolist())
        logger.cur_reward_sum[new_ids] = 0
        logger.cur_episode_length[new_ids] = 0
        if intrinsic_rewards is not None:
            logger.erewbuffer.extend(logger.cur_ereward_sum[new_ids][:, 0].numpy().tolist())
            logger.irewbuffer.extend(logger.cur_ireward_sum[new_ids][:, 0].numpy().tolist())
            logger.cur_ereward_sum[new_ids] = 0
            logger.cur_ireward_sum[new_ids] = 0

    logger.process_env_step = process_env_step_cpu


def _cpu_tensor(value, *, dtype):
    if hasattr(value, "detach"):
        return value.detach().to(device="cpu", dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def _cpu_log_mapping(values) -> dict:
    if not isinstance(values, dict):
        return values
    return {key: _cpu_log_value(value) for key, value in values.items()}


def _cpu_log_value(value):
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        if tensor.numel() == 1:
            return float(tensor.reshape(-1)[0])
        return tensor
    return value
