import hydra
from segdac.agents.agent import Agent
from segdac_dev.replay_buffers.facade import ReplayBufferFacade
import torch
from segdac_dev.logging.loggers.logger import Logger
from comet_ml.exceptions import InterruptedExperiment
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from segdac_dev.envs.factory import create_train_env
from segdac_dev.envs.factory import create_eval_env
from segdac_dev.replay_buffers.factory import create_replay_buffer
from segdac_dev.replay_buffers.segments.factory import (
    create_segmentation_data_replay_buffer,
)
from segdac_dev.trainers.rl_online_trainer import RlOnlineTrainer
from segdac_dev.evaluation.rl_evaluator import RlEvaluator
from segdac.action_scaling.env_action_scaler import TanhEnvActionScaler
from segdac_dev.reproducibility.seed import set_seed
from segdac_dev.logging.artifacts import log_model_weights
from segdac_dev.conversion.numpy_to_torch import convert_dtype
from tensordict import from_module
from ultralytics import settings


@hydra.main(version_base=None, config_path="../configs/", config_name="train_rl_online")
def main(cfg: DictConfig):
    from loguru import logger as console_logger

    settings.update({"sync": False})

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False  # On Ampere+
    torch.backends.cudnn.allow_tf32 = False

    logger: Logger = instantiate(cfg["logger"])

    job_id = logger.get_job_id()

    logger.add_tag(cfg["experiment"]["type"])
    logger.set_name(cfg["experiment"]["name"])
    logger.add_tag(cfg["env"]["name"])
    logger.add_tag(cfg["algo"]["name"])

    logger.log_other("job_id", job_id)
    console_logger.info(f"Job ID: {job_id}")

    set_seed(cfg["training"]["seed"])

    train_env = create_train_env(cfg=cfg, job_id=job_id)
    train_env.set_seed(cfg["training"]["seed"])

    config_updater = instantiate(cfg["algo"]["config_updater"])
    cfg = config_updater.update_config(env=train_env, cfg=cfg)

    logger.log_parameters(OmegaConf.to_container(cfg, resolve=True))
    logger.log_code("./segdac/")
    logger.log_code("./segdac_dev/")
    logger.log_code("./scripts/")
    logger.log_code("./baselines/")
    logger.log_code("./configs/")

    policy_device = torch.device(cfg["policy_device"])

    action_low = train_env.action_space.low
    action_high = train_env.action_space.high
    env_action_scaler = TanhEnvActionScaler(
        action_low=torch.as_tensor(
            action_low, device=policy_device, dtype=convert_dtype(action_low.dtype)
        ),
        action_high=torch.as_tensor(
            action_high, device=policy_device, dtype=convert_dtype(action_high.dtype)
        ),
    )

    console_logger.info(f"Policy Device: {policy_device}")
    train_env_device = cfg["training"]["env_config"]["device"]
    console_logger.info(f"Train Env Device: {train_env_device}")
    eval_env_device = cfg["evaluation"]["env_config"]["device"]
    console_logger.info(f"Eval Env Device: {eval_env_device}")

    agent_train: Agent = (
        instantiate(cfg["algo"]["agent"])(
            env_action_scaler=env_action_scaler,
        )
        .to(policy_device)
        .train()
    )
    agent_train.enable_stochasticity()
    agent_train.disable_exploration()

    agent_eval: Agent = (
        instantiate(cfg["algo"]["agent"])(
            env_action_scaler=env_action_scaler,
        )
        .to(policy_device)
        .eval()
    )
    agent_eval.disable_stochasticity()
    agent_eval.disable_exploration()

    agent_env_collect: Agent = (
        instantiate(cfg["algo"]["agent"])(
            env_action_scaler=env_action_scaler,
        )
        .to(policy_device)
        .eval()
    )
    agent_env_collect.enable_stochasticity()
    agent_env_collect.enable_exploration()

    # Copy params without grad
    from_module(agent_train).data.to_module(agent_env_collect)
    from_module(agent_train).data.to_module(agent_eval)

    agent_env_collect.compile(compile_config=cfg["algo"].get("compile_config", {}))

    data_replay_buffer = create_replay_buffer(cfg, env_action_scaler)

    segmentation_data_replay_buffer = create_segmentation_data_replay_buffer(cfg)

    eval_env = create_eval_env(cfg=cfg, job_id=job_id)
    eval_env.set_seed(cfg["evaluation"]["seed"])

    evaluator = RlEvaluator(
        logger=logger, eval_env=eval_env, cfg=cfg, agent=agent_eval, job_id=job_id
    )

    pre_save_transform = [
        instantiate(pre_save_transform_config)
        for pre_save_transform_config in list(
            cfg["algo"]["replay_buffer"].get("pre_save_transforms", [])
        )
    ]

    replay_buffer = ReplayBufferFacade(
        pre_save_transform=pre_save_transform,
        data_replay_buffer=data_replay_buffer,
        segmentation_data_replay_buffer=segmentation_data_replay_buffer,
    )

    if cfg["algo"].get("agent_needs_rb", False):
        agent_train.replay_buffer = replay_buffer

    trainer = RlOnlineTrainer(
        agent_train=agent_train,
        agent_env_collect=agent_env_collect,
        train_env=train_env,
        evaluator=evaluator,
        replay_buffer=replay_buffer,
        cfg=cfg,
        env_action_scaler=env_action_scaler,
        job_id=job_id,
    )

    try:
        best_agent_file_path, final_agent_file_path = trainer.train()
        logger.add_tag("no_interruption")
    except InterruptedExperiment as exc:
        logger.add_tag("stopped")
        logger.log_other("status", str(exc))
        console_logger.info("Experiment was interrupted.")
    except Exception as ex:
        logger.add_tag("crashed")
        raise ex
    finally:
        if trainer.env_step > 50_000 and cfg["logging"]["save_model"]:
            env_name = cfg["env"]["name"].lower()
            algo_name = cfg["algo"]["name"].lower()
            seed = cfg["training"]["seed"]
            best_model_name = f"{env_name}_{algo_name}_best_{seed}"
            final_model_name = f"{env_name}_{algo_name}_final_{seed}"
            log_model_weights(
                console_logger=console_logger,
                best_model_file_path=best_agent_file_path,
                final_model_file_path=final_agent_file_path,
                logger=logger,
                best_model_name=best_model_name,
                final_model_name=final_model_name,
            )
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
