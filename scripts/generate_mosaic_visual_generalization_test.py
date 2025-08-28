import hydra
import math
import copy
from segdac_dev.logging.loggers.logger import Logger
from omegaconf import open_dict, DictConfig, OmegaConf
from hydra.utils import instantiate
from segdac_dev.envs.factory import create_test_env
from segdac_dev.reproducibility.seed import set_seed
from pathlib import Path
from torchvision.utils import save_image
from PIL import Image
from ultralytics import settings


def recursive_override(src: dict, dest: dict):
    for key, src_val in src.items():
        dest_val = dest[key]
        if isinstance(src_val, dict) and isinstance(dest_val, dict):
            recursive_override(src_val, dest_val)
        else:
            dest[key] = src_val


def make_image_grid(paths, rows, cols, cell_size=None, output_path="grid.jpg"):
    if cell_size is None:
        with Image.open(paths[0]) as im:
            cell_w, cell_h = im.size
    else:
        cell_w, cell_h = cell_size
    grid_w, grid_h = cols * cell_w, rows * cell_h
    grid = Image.new("RGB", (grid_w, grid_h))
    for idx, path in enumerate(paths):
        if idx >= rows * cols:
            raise Exception("Number of images exceeds rows x cols!")
        with Image.open(path) as im:
            if im.size != (cell_w, cell_h):
                im = im.resize((cell_w, cell_h))
            x = (idx % cols) * cell_w
            y = (idx // cols) * cell_h
            grid.paste(im, (x, y))
    grid.save(output_path)


@hydra.main(version_base=None, config_path="../configs/", config_name="test_visual_generalization")
def main(cfg: DictConfig):
    from loguru import logger as console_logger
    settings.update({"sync": False})

    logger: Logger = instantiate(cfg["logger"])
    job_id = logger.get_job_id()
    individual_images_folder_path = (
        Path(cfg["final_job_data_dir"])
        / job_id
        / cfg["logging"]["images_dir"]
        / "individual_images"
    )
    individual_images_folder_path.mkdir(parents=True, exist_ok=True)
    logger.log_other("job_id", job_id)
    console_logger.info(f"Job ID: {job_id}")

    env_configs = [cfg[f"env_{i}"] for i in range(1, 9)]
    for env in env_configs:
        env["name"] = f"maniskill3_{env['id']}"
        logger.add_tag(env["name"])

    logger.log_parameters(OmegaConf.to_container(cfg, resolve=False))
    logger.log_code("./segdac/")
    logger.log_code("./segdac_dev/")
    logger.log_code("./scripts/")
    logger.log_code("./baselines/")
    logger.log_code("./configs/")

    set_seed(cfg["evaluation"]["seed"])

    image_paths = []
    difficulties = ["easy", "medium", "hard"]

    for env_config in env_configs:
        task_id = env_config["id"]
        task_key = task_id.lower()
        console_logger.info(f"Task: {task_key}")
        env_image_paths = []

        cfg_copy = copy.deepcopy(cfg)
        with open_dict(cfg_copy):
            cfg_copy.env = env_config
        test_config = OmegaConf.to_container(env_config['default_config'])
        test_env = create_test_env(
            cfg=cfg_copy, job_id=job_id, test_config=test_config)
        test_env.set_seed(cfg["evaluation"]["seed"])
        img = test_env.reset().data.to("cpu", non_blocking=False)[
            "pixels"][0][0] / 255.0
        p0 = individual_images_folder_path / \
            f"{task_key}_no_perturbation_test.jpg"
        save_image(img, p0)
        image_paths.append(p0)
        test_env.close()

        for difficulty in difficulties:
            console_logger.info(f"Difficulty : {difficulty}")

            test_difficulty_specific_config = OmegaConf.to_container(
                env_config[difficulty], resolve=True)

            env_image_paths.append(p0)
            nb_cols = 1

            for test_type, test_specific_overrides in test_difficulty_specific_config.items():
                console_logger.info(f"Perturbation Test : {test_type}")
                cfg_copy = copy.deepcopy(cfg)
                with open_dict(cfg_copy):
                    cfg_copy.env = env_config
                test_config = OmegaConf.to_container(
                    env_config['default_config'])
                recursive_override(test_specific_overrides, test_config)
                test_env = create_test_env(
                    cfg=cfg_copy, job_id=job_id, test_config=test_config)
                test_env.set_seed(cfg["evaluation"]["seed"])

                img2 = test_env.reset().data.to("cpu", non_blocking=False)[
                    "pixels"][0][0] / 255.0
                p2 = individual_images_folder_path / \
                    f"{task_key}_{test_type}_{difficulty}.jpg"
                image_paths.append(p2)
                env_image_paths.append(p2)
                save_image(img2, p2)
                test_env.close()

                nb_cols += 1

        mosaic_output_path = (
            Path(cfg["final_job_data_dir"])
            / job_id
            / cfg["logging"]["images_dir"]
            / f"mosaic_{task_key}.jpg"
        )
        make_image_grid(
            env_image_paths,
            rows=len(difficulties),
            cols=nb_cols,
            output_path=mosaic_output_path,
        )

    mosaic_output_path = (
        Path(cfg["final_job_data_dir"])
        / job_id
        / cfg["logging"]["images_dir"]
        / "mosaic_all.jpg"
    )

    total = len(image_paths)
    nb_rows = int(math.ceil(math.sqrt(total)))
    nb_cols = int(math.ceil(total / nb_rows))

    make_image_grid(
        image_paths,
        rows=nb_rows,
        cols=nb_cols,
        output_path=mosaic_output_path,
    )


if __name__ == "__main__":
    main()
