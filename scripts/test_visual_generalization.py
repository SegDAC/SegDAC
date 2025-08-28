import json
import hydra
import torch
import copy
import sys
import numpy as np
from segdac_dev.logging.loggers.logger import Logger
from comet_ml.exceptions import InterruptedExperiment
from omegaconf import open_dict
from segdac.agents.agent import Agent
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from segdac_dev.envs.factory import create_test_env
from segdac.action_scaling.env_action_scaler import TanhEnvActionScaler
from segdac_dev.reproducibility.seed import set_seed
from segdac_dev.conversion.numpy_to_torch import convert_dtype
from segdac_dev.testing.agent_tester import AgentTester
from pathlib import Path
from rliable import library as rly
from rliable import metrics
from ultralytics import settings


def get_scores(
    data: dict,
    m: int,
    tasks: list[str],
    perturbation_tests: list[str],
    difficulties: list[str],
    metric_name: str,
    nb_seeds: int = 5,
) -> np.ndarray:
    n = nb_seeds
    scores = np.zeros((m, n), dtype=np.float32)
    m_index = 0

    for task_key, perturbation_tests_val in data.items():
        if task_key in tasks:
            for perturbation_test_key, difficulty_val in perturbation_tests_val.items():
                if perturbation_test_key in perturbation_tests:
                    for difficulty in difficulties:
                        scores[m_index] = np.array(
                            difficulty_val[difficulty][metric_name]
                        )
                        m_index += 1

    return scores


def get_no_perturb_scores(
    data: dict,
    m: int,
    tasks: list[str],
    metric_name: str,
    nb_seeds: int = 5
) -> np.ndarray:
    n = nb_seeds
    scores = np.zeros((m, n), dtype=np.float32)
    m_index = 0

    for task in tasks:
        task_scores = data[task]["no_perturbation_test"][metric_name]
        scores[m_index] = np.array(task_scores)
        m_index += 1

    return scores


def recursive_override(src: dict, dest: dict):
    for key, src_val in src.items():
        dest_val = dest[key]
        if isinstance(src_val, dict) and isinstance(dest_val, dict):
            recursive_override(src_val, dest_val)
        else:
            dest[key] = src_val


@hydra.main(version_base=None, config_path="../configs/", config_name="test_visual_generalization")
def main(cfg: DictConfig):
    from loguru import logger as console_logger
    settings.update({"sync": False})

    LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"

    console_logger.configure(
        handlers=[
            {"sink": sys.stderr, "level": "INFO",
                "format": LOG_FORMAT, "colorize": True},
            {"sink": str((Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / Path("test_visual_generalization.log")).resolve()),
             "level": "DEBUG", "format": LOG_FORMAT, "colorize": False},
        ]
    )

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False  # On Ampere+
    torch.backends.cudnn.allow_tf32 = False

    logger: Logger = instantiate(cfg["logger"])

    job_id = logger.get_job_id()

    assets_folder_path = Path(
        cfg["final_job_data_dir"]) / Path(job_id) / Path(cfg["logging"]["assets_dir"])
    assets_folder_path.mkdir(parents=True, exist_ok=True)
    logger.log_other("job_id", job_id)
    console_logger.info(f"Job ID: {job_id}")

    logger.add_tag(cfg["experiment"]["type"])
    logger.set_name(cfg["experiment"]["name"])
    algo_name = cfg["algo_1"]["name"]
    console_logger.info(f"Algo : {algo_name}")
    cfg['env_1']['name'] = f"maniskill3_{cfg['env_1']['id']}"
    cfg['env_2']['name'] = f"maniskill3_{cfg['env_2']['id']}"
    cfg['env_3']['name'] = f"maniskill3_{cfg['env_3']['id']}"
    cfg['env_4']['name'] = f"maniskill3_{cfg['env_4']['id']}"
    cfg['env_5']['name'] = f"maniskill3_{cfg['env_5']['id']}"
    cfg['env_6']['name'] = f"maniskill3_{cfg['env_6']['id']}"
    cfg['env_7']['name'] = f"maniskill3_{cfg['env_7']['id']}"
    cfg['env_8']['name'] = f"maniskill3_{cfg['env_8']['id']}"
    env_configs = [
        cfg["env_1"],
        cfg["env_2"],
        cfg["env_3"],
        cfg["env_4"],
        cfg["env_5"],
        cfg["env_6"],
        cfg["env_7"],
        cfg["env_8"],
    ]
    env_names = [env_config["name"] for env_config in env_configs]
    for env_name in env_names:
        logger.add_tag(env_name)

    logger.add_tag(algo_name)

    logger.log_parameters(OmegaConf.to_container(cfg, resolve=False))

    logger.log_code("./segdac/")
    logger.log_code("./segdac_dev/")
    logger.log_code("./scripts/")
    logger.log_code("./baselines/")
    logger.log_code("./configs/")

    seed = cfg["evaluation"]["seed"]
    set_seed(seed)

    difficulties = ["easy", "medium", "hard"]
    policy_device = torch.device(cfg["policy_device"])
    console_logger.info(f"Policy Device: {policy_device}")
    test_env_device = cfg["evaluation"]["env_config"]["device"]
    console_logger.info(f"Env Device: {test_env_device}")
    num_rollouts = cfg['evaluation']['num_rollouts']
    agent_tester = AgentTester(
        logger=logger,
        cfg=cfg
    )
    config_updater = instantiate(cfg["algo_1"]["config_updater"])
    nb_seeds = len(list(cfg["agent_weights"]['env_1']))
    logger.log_other("nb_seeds", nb_seeds)
    metrics_data = {
        algo_name: {

        }
    }
    try:
        for env_id, env_config in enumerate(env_configs):
            task_id = env_config["id"]
            console_logger.info(f"Task : {task_id}")

            env_max_frames_per_traj = int(env_config['max_frames_per_traj'])
            action_repeat = int(env_config['action_repeat'])

            agent_weights_paths = list(cfg['agent_weights'][f"env_{env_id+1}"])

            for agent_id, agent_weights_path in enumerate(agent_weights_paths):
                console_logger.info(f"Model Seed Index : {agent_id}")

                console_logger.info("No Perturbation Baseline")

                cfg_copy = copy.deepcopy(cfg)
                with open_dict(cfg_copy):
                    cfg_copy.env = env_config
                    cfg_copy.algo = cfg_copy[f"algo_{env_id+1}"]
                test_config = OmegaConf.to_container(
                    env_config['default_config'])
                test_env = create_test_env(
                    cfg=cfg_copy, job_id=job_id, test_config=test_config)
                test_env.set_seed(cfg["evaluation"]["seed"])
                cfg_copy = config_updater.update_config(
                    env=test_env, cfg=cfg_copy)

                action_low = test_env.action_space.low
                action_high = test_env.action_space.high
                env_action_scaler = TanhEnvActionScaler(
                    action_low=torch.as_tensor(
                        action_low, device=policy_device, dtype=convert_dtype(
                            action_low.dtype)
                    ),
                    action_high=torch.as_tensor(
                        action_high, device=policy_device, dtype=convert_dtype(
                            action_high.dtype)
                    ),
                )
                agent: Agent = (
                    instantiate(cfg_copy["algo"]["agent"])(
                        env_action_scaler=env_action_scaler,
                    )
                    .eval()
                ).to(policy_device)
                agent.disable_stochasticity()
                agent.disable_exploration()

                console_logger.info("Loading agent model weights...")
                agent_state_dict = torch.load(
                    agent_weights_path, map_location=policy_device, weights_only=True)
                agent.load_state_dict(
                    agent_state_dict,
                    strict=True,
                )
                console_logger.success("Agent model weights loaded!")
                agent = agent.to(policy_device).eval()

                task_id_formatted = task_id.lower()
                test_type = "no_perturbation_test"
                logging_prefix = f"{task_id_formatted}_{test_type}_"

                env_seed_no_perturb_test_metrics = agent_tester.test(
                    agent=agent,
                    test_env=test_env,
                    num_rollouts=num_rollouts,
                    logging_prefix=logging_prefix,
                    seed_number=agent_id,
                    env_max_frames_per_traj=env_max_frames_per_traj,
                    action_repeat=action_repeat
                )
                test_env.close()

                del test_env

                torch.cuda.empty_cache()

                baseline_return_score_normalized = env_seed_no_perturb_test_metrics[
                    f"{logging_prefix}return"] / env_max_frames_per_traj
                baseline_success_at_end_score_normalized = env_seed_no_perturb_test_metrics[
                    f"{logging_prefix}success_at_end_mean"]

                if metrics_data[algo_name].get(task_id_formatted, None) is None:
                    metrics_data[algo_name][task_id_formatted] = {
                        test_type: {
                            "return": [baseline_return_score_normalized],
                            "success_at_end": [baseline_success_at_end_score_normalized]
                        }
                    }
                else:
                    metrics_data[algo_name][task_id_formatted][test_type]["return"].append(
                        baseline_return_score_normalized)
                    metrics_data[algo_name][task_id_formatted][test_type]["success_at_end"].append(
                        baseline_success_at_end_score_normalized)

                for difficulty in difficulties:
                    console_logger.info(f"Difficulty : {difficulty}")

                    test_difficulty_specific_config = OmegaConf.to_container(
                        env_config[difficulty], resolve=True)

                    for test_type, test_specific_overrides in test_difficulty_specific_config.items():
                        console_logger.info(f"Perturbation Test : {test_type}")
                        cfg_copy = copy.deepcopy(cfg)
                        with open_dict(cfg_copy):
                            cfg_copy.env = env_config
                            cfg_copy.algo = cfg_copy[f"algo_{env_id+1}"]
                        test_config = OmegaConf.to_container(
                            env_config['default_config'])
                        recursive_override(
                            test_specific_overrides, test_config)
                        test_env = create_test_env(
                            cfg=cfg_copy, job_id=job_id, test_config=test_config)
                        test_env.set_seed(cfg["evaluation"]["seed"])
                        cfg_copy = config_updater.update_config(
                            env=test_env, cfg=cfg_copy)

                        task_id_formatted = task_id.lower()
                        logging_prefix = f"{task_id_formatted}_{test_type}_{difficulty}_"
                        env_max_frames_per_traj = int(
                            env_config['max_frames_per_traj'])
                        env_seed_diff_test_metrics = agent_tester.test(
                            agent=agent,
                            test_env=test_env,
                            num_rollouts=num_rollouts,
                            logging_prefix=logging_prefix,
                            seed_number=agent_id,
                            env_max_frames_per_traj=env_max_frames_per_traj,
                            action_repeat=int(env_config['action_repeat'])
                        )
                        test_env.close()

                        del test_env

                        torch.cuda.empty_cache()

                        new_return_score_normalized = env_seed_diff_test_metrics[
                            f"{logging_prefix}return"] / env_max_frames_per_traj
                        new_success_at_end_score_normalized = env_seed_diff_test_metrics[
                            f"{logging_prefix}success_at_end_mean"]

                        if metrics_data[algo_name].get(task_id_formatted, None) is None:
                            metrics_data[algo_name][task_id_formatted] = {
                                test_type: {
                                    difficulty: {
                                        "return": [new_return_score_normalized],
                                        "success_at_end": [new_success_at_end_score_normalized],
                                    }
                                }
                            }
                        elif metrics_data[algo_name].get(task_id_formatted, {}).get(test_type, None) is None:
                            metrics_data[algo_name][task_id_formatted][test_type] = {
                                difficulty: {
                                    "return": [new_return_score_normalized],
                                    "success_at_end": [new_success_at_end_score_normalized],
                                }
                            }
                        elif metrics_data[algo_name].get(task_id_formatted, {}).get(test_type, {}).get(difficulty, None) is None:
                            metrics_data[algo_name][task_id_formatted][test_type][difficulty] = {
                                "return": [new_return_score_normalized],
                                "success_at_end": [new_success_at_end_score_normalized],
                            }
                        else:
                            metrics_data[algo_name][task_id_formatted][test_type][difficulty]["return"].append(
                                new_return_score_normalized)
                            metrics_data[algo_name][task_id_formatted][test_type][difficulty]["success_at_end"].append(
                                new_success_at_end_score_normalized)

        console_logger.info("Saving raw test scores...")
        test_results_path = assets_folder_path / \
            Path(f"{algo_name}_test_results.json")
        with open(test_results_path, 'w') as f:
            json.dump(metrics_data, f)
        logger.log_asset(str(test_results_path.resolve()), step=0)

        console_logger.info("Aggregating scores...")

        camera_perturbation_tests = ['camera_pose_test', 'camera_fov_test']
        lighting_perturbation_tests = [
            'lighting_direction_test', 'lighting_color_test']
        color_perturbation_tests = [
            'mo_color_test', 'ro_color_test', 'table_color_test', 'ground_color_test']
        texture_perturbation_tests = [
            'mo_texture_test', 'ro_texture_test', 'table_texture_test', 'ground_texture_test']
        all_perturbation_tests = camera_perturbation_tests + lighting_perturbation_tests + \
            color_perturbation_tests + texture_perturbation_tests

        assert len(all_perturbation_tests) == 12

        ro_tests = ['ro_color_test', 'ro_texture_test']

        task_perturbation_tests = {
            'pushcubetest-v1': all_perturbation_tests,
            'pullcubetest-v1': all_perturbation_tests,
            # This env doesn't have RO
            'pickcubetest-v1': set(all_perturbation_tests) - set(ro_tests),
            'pokecubetest-v1': all_perturbation_tests,
            'pullcubetooltest-v1': all_perturbation_tests,
            # This env doesn't have RO
            'liftpeguprighttest-v1': set(all_perturbation_tests) - set(ro_tests),
            'unitreeg1placeappleinbowltest-v1': all_perturbation_tests,
            # This env doesn't have RO
            'unitreeg1transportboxtest-v1': set(all_perturbation_tests) - set(ro_tests),
        }

        task_ids = [c['id'].lower() for c in env_configs]
        reps = cfg['metrics_aggregation']['reps']
        confidence_interval_size = cfg['metrics_aggregation']['confidence_interval_size']
        random_state = np.random.RandomState(seed=seed)
        algo_aggregated_scores = {
            algo_name: {
                "no_perturb": {

                },
                "perturb_per_diff_per_cat": {

                },
                "perturb_indiv": {

                }
            }
        }
        m = len(task_ids)
        console_logger.info("Computing overall-no_perturbation_test scores...")
        no_perturb_overall_return_scores = get_no_perturb_scores(
            data=metrics_data[algo_name],
            m=m,
            tasks=task_ids,
            nb_seeds=nb_seeds,
            metric_name="return"
        )
        no_perturb_overall_success_at_end_scores = get_no_perturb_scores(
            data=metrics_data[algo_name],
            m=m,
            tasks=task_ids,
            nb_seeds=nb_seeds,
            metric_name="success_at_end"
        )
        agg_no_perturb_overall_return_scores, agg_no_perturb_overall_return_score_cis = rly.get_interval_estimates(
            {algo_name: no_perturb_overall_return_scores},
            lambda x: np.array([metrics.aggregate_iqm(x)]),
            reps=reps,
            confidence_interval_size=confidence_interval_size,
            random_state=random_state
        )
        agg_no_perturb_overall_success_at_end_scores, agg_no_perturb_overall_success_at_end_score_cis = rly.get_interval_estimates(
            {algo_name: no_perturb_overall_success_at_end_scores},
            lambda x: np.array([metrics.aggregate_iqm(x)]),
            reps=reps,
            confidence_interval_size=confidence_interval_size,
            random_state=random_state
        )
        algo_aggregated_scores[algo_name]["no_perturb"]["overall"] = {
            "return": {
                'iqm': agg_no_perturb_overall_return_scores[algo_name].tolist(),
                'ci': agg_no_perturb_overall_return_score_cis[algo_name].tolist()
            },
            "success_at_end": {
                'iqm': agg_no_perturb_overall_success_at_end_scores[algo_name].tolist(),
                'ci': agg_no_perturb_overall_success_at_end_score_cis[algo_name].tolist()
            }
        }
        for difficulty in difficulties:
            algo_aggregated_scores[algo_name]["perturb_indiv"][difficulty] = {}
            difficulty_list = [difficulty]
            camera_m = 0
            lighting_m = 0
            color_m = 0
            texture_m = 0
            for task_id in task_ids:
                task_perturbations = set(task_perturbation_tests[task_id])
                camera_m += len(list(task_perturbations &
                                set(camera_perturbation_tests)))
                lighting_m += len(list(task_perturbations &
                                  set(lighting_perturbation_tests)))
                color_m += len(list(task_perturbations &
                               set(color_perturbation_tests)))
                texture_m += len(list(task_perturbations &
                                 set(texture_perturbation_tests)))

            overall_m = camera_m + lighting_m + color_m + texture_m

            console_logger.info(f"Difficulty: {difficulty}")
            console_logger.info(f"overall_m: {overall_m}")
            console_logger.info(f"camera_m: {camera_m}")
            console_logger.info(f"lighting_m: {lighting_m}")
            console_logger.info(f"color_m: {color_m}")
            console_logger.info(f"texture_m: {texture_m}")

            console_logger.info("Computing Overall scores...")
            overall_return_scores = get_scores(
                metrics_data[algo_name],
                m=overall_m,
                tasks=task_ids,
                perturbation_tests=all_perturbation_tests,
                difficulties=difficulty_list,
                nb_seeds=nb_seeds,
                metric_name="return"
            )
            overall_agg_return_scores, overall_agg_return_score_cis = rly.get_interval_estimates(
                {algo_name: overall_return_scores},
                lambda x: np.array([metrics.aggregate_iqm(x)]),
                reps=reps,
                confidence_interval_size=confidence_interval_size,
                random_state=random_state
            )
            overall_success_at_end_scores = get_scores(
                metrics_data[algo_name],
                m=overall_m,
                tasks=task_ids,
                perturbation_tests=all_perturbation_tests,
                difficulties=difficulty_list,
                nb_seeds=nb_seeds,
                metric_name="success_at_end"
            )
            overall_agg_success_at_end_scores, overall_agg_success_at_end_score_cis = rly.get_interval_estimates(
                {algo_name: overall_success_at_end_scores},
                lambda x: np.array([metrics.aggregate_iqm(x)]),
                reps=reps,
                confidence_interval_size=confidence_interval_size,
                random_state=random_state
            )
            console_logger.info("Computing Camera scores...")
            camera_return_scores = get_scores(
                metrics_data[algo_name],
                m=camera_m,
                tasks=task_ids,
                perturbation_tests=camera_perturbation_tests,
                difficulties=difficulty_list,
                nb_seeds=nb_seeds,
                metric_name="return"
            )
            camera_success_at_end_scores = get_scores(
                metrics_data[algo_name],
                m=camera_m,
                tasks=task_ids,
                perturbation_tests=camera_perturbation_tests,
                difficulties=difficulty_list,
                nb_seeds=nb_seeds,
                metric_name="success_at_end"
            )
            camera_agg_return_scores, camera_agg_return_score_cis = rly.get_interval_estimates(
                {algo_name: camera_return_scores},
                lambda x: np.array([metrics.aggregate_iqm(x)]),
                reps=reps,
                confidence_interval_size=confidence_interval_size,
                random_state=random_state
            )
            camera_agg_success_at_end_scores, camera_agg_success_at_end_score_cis = rly.get_interval_estimates(
                {algo_name: camera_success_at_end_scores},
                lambda x: np.array([metrics.aggregate_iqm(x)]),
                reps=reps,
                confidence_interval_size=confidence_interval_size,
                random_state=random_state
            )
            console_logger.info("Computing Lighting scores...")
            lighting_return_scores = get_scores(
                metrics_data[algo_name],
                m=lighting_m,
                tasks=task_ids,
                perturbation_tests=lighting_perturbation_tests,
                difficulties=difficulty_list,
                nb_seeds=nb_seeds,
                metric_name="return"
            )
            lighting_success_at_end_scores = get_scores(
                metrics_data[algo_name],
                m=lighting_m,
                tasks=task_ids,
                perturbation_tests=lighting_perturbation_tests,
                difficulties=difficulty_list,
                nb_seeds=nb_seeds,
                metric_name="success_at_end"
            )
            lighting_agg_return_scores, lighting_agg_return_score_cis = rly.get_interval_estimates(
                {algo_name: lighting_return_scores},
                lambda x: np.array([metrics.aggregate_iqm(x)]),
                reps=reps,
                confidence_interval_size=confidence_interval_size,
                random_state=random_state
            )
            lighting_agg_success_at_end_scores, lighting_agg_success_at_end_score_cis = rly.get_interval_estimates(
                {algo_name: lighting_success_at_end_scores},
                lambda x: np.array([metrics.aggregate_iqm(x)]),
                reps=reps,
                confidence_interval_size=confidence_interval_size,
                random_state=random_state
            )
            console_logger.info("Computing Color scores...")
            color_return_scores = get_scores(
                metrics_data[algo_name],
                m=color_m,
                tasks=task_ids,
                perturbation_tests=color_perturbation_tests,
                difficulties=difficulty_list,
                nb_seeds=nb_seeds,
                metric_name="return"
            )
            color_success_at_end_scores = get_scores(
                metrics_data[algo_name],
                m=color_m,
                tasks=task_ids,
                perturbation_tests=color_perturbation_tests,
                difficulties=difficulty_list,
                nb_seeds=nb_seeds,
                metric_name="success_at_end"
            )
            color_agg_return_scores, color_agg_return_score_cis = rly.get_interval_estimates(
                {algo_name: color_return_scores},
                lambda x: np.array([metrics.aggregate_iqm(x)]),
                reps=reps,
                confidence_interval_size=confidence_interval_size,
                random_state=random_state
            )
            color_agg_success_at_end_scores, color_agg_success_at_end_score_cis = rly.get_interval_estimates(
                {algo_name: color_success_at_end_scores},
                lambda x: np.array([metrics.aggregate_iqm(x)]),
                reps=reps,
                confidence_interval_size=confidence_interval_size,
                random_state=random_state
            )
            console_logger.info("Computing Texture scores...")
            texture_return_scores = get_scores(
                metrics_data[algo_name],
                m=texture_m,
                tasks=task_ids,
                perturbation_tests=texture_perturbation_tests,
                difficulties=difficulty_list,
                nb_seeds=nb_seeds,
                metric_name="return"
            )
            texture_success_at_end_scores = get_scores(
                metrics_data[algo_name],
                m=texture_m,
                tasks=task_ids,
                perturbation_tests=texture_perturbation_tests,
                difficulties=difficulty_list,
                nb_seeds=nb_seeds,
                metric_name="success_at_end"
            )
            texture_agg_return_scores, texture_agg_return_score_cis = rly.get_interval_estimates(
                {algo_name: texture_return_scores},
                lambda x: np.array([metrics.aggregate_iqm(x)]),
                reps=reps,
                confidence_interval_size=confidence_interval_size,
                random_state=random_state
            )
            texture_agg_success_at_end_scores, texture_agg_success_at_end_score_cis = rly.get_interval_estimates(
                {algo_name: texture_success_at_end_scores},
                lambda x: np.array([metrics.aggregate_iqm(x)]),
                reps=reps,
                confidence_interval_size=confidence_interval_size,
                random_state=random_state
            )
            algo_aggregated_scores[algo_name]["perturb_per_diff_per_cat"][difficulty] = {
                'overall': {
                    "return": {
                        'iqm': overall_agg_return_scores[algo_name].tolist(),
                        'ci': overall_agg_return_score_cis[algo_name].tolist()
                    },
                    "success_at_end": {
                        'iqm': overall_agg_success_at_end_scores[algo_name].tolist(),
                        'ci': overall_agg_success_at_end_score_cis[algo_name].tolist()
                    }
                },
                'camera': {
                    "return": {
                        'iqm': camera_agg_return_scores[algo_name].tolist(),
                        'ci': camera_agg_return_score_cis[algo_name].tolist()
                    },
                    "success_at_end": {
                        'iqm': camera_agg_success_at_end_scores[algo_name].tolist(),
                        'ci': camera_agg_success_at_end_score_cis[algo_name].tolist()
                    }
                },
                'lighting': {
                    "return": {
                        'iqm': lighting_agg_return_scores[algo_name].tolist(),
                        'ci': lighting_agg_return_score_cis[algo_name].tolist()
                    },
                    "success_at_end": {
                        'iqm': lighting_agg_success_at_end_scores[algo_name].tolist(),
                        'ci': lighting_agg_success_at_end_score_cis[algo_name].tolist()
                    }
                },
                'color': {
                    "return": {
                        'iqm': color_agg_return_scores[algo_name].tolist(),
                        'ci': color_agg_return_score_cis[algo_name].tolist()
                    },
                    "success_at_end": {
                        'iqm': color_agg_success_at_end_scores[algo_name].tolist(),
                        'ci': color_agg_success_at_end_score_cis[algo_name].tolist()
                    }
                },
                "texture": {
                    "return": {
                        'iqm': texture_agg_return_scores[algo_name].tolist(),
                        'ci': texture_agg_return_score_cis[algo_name].tolist()
                    },
                    "success_at_end": {
                        'iqm': texture_agg_success_at_end_scores[algo_name].tolist(),
                        'ci': texture_agg_success_at_end_score_cis[algo_name].tolist()
                    }
                }
            }

            for task_id in task_ids:
                algo_aggregated_scores[algo_name]["perturb_indiv"][difficulty][task_id] = {
                }

                m = 1
                console_logger.info(
                    f"Computing {task_id}-no_perturbation_test scores...")
                no_perturb_indiv_return_scores = get_no_perturb_scores(
                    data=metrics_data[algo_name],
                    m=m,
                    tasks=[task_id],
                    nb_seeds=nb_seeds,
                    metric_name="return"
                )
                no_perturb_indiv_success_at_end_scores = get_no_perturb_scores(
                    data=metrics_data[algo_name],
                    m=m,
                    tasks=[task_id],
                    nb_seeds=nb_seeds,
                    metric_name="success_at_end"
                )

                agg_no_perturb_indiv_return_scores, agg_no_perturb_indiv_return_score_cis = rly.get_interval_estimates(
                    {algo_name: no_perturb_indiv_return_scores.T},
                    lambda x: np.array([metrics.aggregate_iqm(x)]),
                    reps=reps,
                    confidence_interval_size=confidence_interval_size,
                    random_state=random_state
                )
                agg_no_perturb_indiv_success_at_end_scores, agg_no_perturb_indiv_success_at_end_score_cis = rly.get_interval_estimates(
                    {algo_name: no_perturb_indiv_success_at_end_scores.T},
                    lambda x: np.array([metrics.aggregate_iqm(x)]),
                    reps=reps,
                    confidence_interval_size=confidence_interval_size,
                    random_state=random_state
                )
                algo_aggregated_scores[algo_name]["no_perturb"][task_id] = {
                    "return": {
                        'iqm': agg_no_perturb_indiv_return_scores[algo_name].tolist(),
                        'ci': agg_no_perturb_indiv_return_score_cis[algo_name].tolist()
                    },
                    "success_at_end": {
                        'iqm': agg_no_perturb_indiv_success_at_end_scores[algo_name].tolist(),
                        'ci': agg_no_perturb_indiv_success_at_end_score_cis[algo_name].tolist()
                    }
                }

                for perturbation_test in task_perturbation_tests[task_id]:
                    m = 1
                    console_logger.info(
                        f"Computing {task_id}-{perturbation_test}-{difficulty} scores...")
                    perturb_indiv_return_scores = get_scores(
                        metrics_data[algo_name],
                        m=m,
                        tasks=[task_id],
                        perturbation_tests=[perturbation_test],
                        difficulties=difficulty_list,
                        nb_seeds=nb_seeds,
                        metric_name="return"
                    )
                    perturb_indiv_success_at_end_scores = get_scores(
                        metrics_data[algo_name],
                        m=m,
                        tasks=[task_id],
                        perturbation_tests=[perturbation_test],
                        difficulties=difficulty_list,
                        nb_seeds=nb_seeds,
                        metric_name="success_at_end"
                    )

                    agg_perturb_indiv_return_scores, agg_perturb_indiv_return_score_cis = rly.get_interval_estimates(
                        {algo_name: perturb_indiv_return_scores.T},
                        lambda x: np.array([metrics.aggregate_iqm(x)]),
                        reps=reps,
                        confidence_interval_size=confidence_interval_size,
                        random_state=random_state
                    )
                    agg_perturb_indiv_success_at_end_scores, agg_perturb_indiv_success_at_end_score_cis = rly.get_interval_estimates(
                        {algo_name: perturb_indiv_success_at_end_scores.T},
                        lambda x: np.array([metrics.aggregate_iqm(x)]),
                        reps=reps,
                        confidence_interval_size=confidence_interval_size,
                        random_state=random_state
                    )

                    algo_aggregated_scores[algo_name]["perturb_indiv"][difficulty][task_id][perturbation_test] = {
                        "return": {
                            'iqm': agg_perturb_indiv_return_scores[algo_name].tolist(),
                            'ci': agg_perturb_indiv_return_score_cis[algo_name].tolist()
                        },
                        "success_at_end": {
                            'iqm': agg_perturb_indiv_success_at_end_scores[algo_name].tolist(),
                            'ci': agg_perturb_indiv_success_at_end_score_cis[algo_name].tolist()
                        }
                    }

        console_logger.info("Saving aggregated scores...")
        algo_aggregated_scores_path = assets_folder_path / \
            Path(f"{algo_name}_aggregated_scores.json")
        with open(algo_aggregated_scores_path, 'w') as f:
            json.dump(algo_aggregated_scores, f)
        logger.log_asset(str(algo_aggregated_scores_path.resolve()), step=0)

        console_logger.success("Benchmark completed with success!")
        logger.add_tag("no_interruption")
    except InterruptedExperiment as exc:
        logger.add_tag("stopped")
        logger.log_other("status", str(exc))
        console_logger.info("Experiment was interrupted.")
    except Exception as ex:
        logger.add_tag("crashed")
        console_logger.error("Crash occurred!")
        raise ex


if __name__ == "__main__":
    main()
