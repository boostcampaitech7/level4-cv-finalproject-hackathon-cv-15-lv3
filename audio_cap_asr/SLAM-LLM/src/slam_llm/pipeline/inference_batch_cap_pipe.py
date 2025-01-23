# import fire
import random
import torch
import logging
# import argparse
from slam_llm.models.slam_model import slam_model
# config
# from llama_recipes.configs import fsdp_config as FSDP_CONFIG
# from llama_recipes.configs import train_config as TRAIN_CONFIG
# from llama_recipes.configs import model_config as MODEL_CONFIG
# from llama_recipes.configs import log_config as LOG_CONFIG

from slam_llm.utils.model_utils import get_custom_model_factory
from slam_llm.utils.dataset_utils import get_preprocessed_dataset
import os
import logging
from tqdm import tqdm

import json
import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf


#@hydra.main(config_name=None, version_base=None)
@hydra.main(config_path=None, config_name=None)
def main_hydra(cfg: DictConfig):
	def to_plain_list(cfg_item):
		if isinstance(cfg_item, ListConfig):
			return OmegaConf.to_container(cfg_item, resolve=True)
		elif isinstance(cfg_item, DictConfig):
			return {k: to_plain_list(v) for k, v in cfg_item.items()}
		else:
			return cfg_item
	
	# kwargs = to_plain_list(cfg)
	kwargs = cfg
	log_level = getattr(logging, kwargs.get("log_level", "INFO").upper())
	
	logging.basicConfig(level=log_level)
	
	if kwargs.get("debug", False):
		import pdb;
		pdb.set_trace()
	
	main(kwargs)


def main(kwargs: DictConfig):
    train_config, fsdp_config, model_config, log_config, dataset_config = (
        kwargs.train_config,
        kwargs.fsdp_config,
        kwargs.model_config,
        kwargs.log_config,
        kwargs.dataset_config,
    )

    OmegaConf.set_struct(kwargs, False)
    del kwargs["train_config"]
    del kwargs["fsdp_config"]
    del kwargs["model_config"]
    del kwargs["log_config"]
    del kwargs["dataset_config"]
    OmegaConf.set_struct(kwargs, True)

    # Set up logging
    if not os.path.exists(os.path.dirname(log_config.log_file)):
        os.makedirs(os.path.dirname(log_config.log_file), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )

    # Initialize model
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    model_factory = get_custom_model_factory(model_config, logger)
    model, tokenizer = model_factory(train_config, model_config, **kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset_test = get_preprocessed_dataset(tokenizer, dataset_config, split="test")
    test_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        shuffle=False,
        batch_size=train_config.val_batch_size,
        drop_last=False,
        collate_fn=dataset_test.collator,
    )

    # Handle JSON for decode_log
    decode_log_path = kwargs.get("decode_log")
    if os.path.exists(decode_log_path):
        with open(decode_log_path, "r") as json_file:
            decode_log_data = json.load(json_file)
    else:
        decode_log_data = {}

    logger.info("=====================================")

    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        for key in batch.keys():
            if isinstance(batch[key], torch.Tensor):
                logger.info(f"Batch key: {key}, dtype: {batch[key].dtype}")
            batch[key] = batch[key].to(device) if isinstance(batch[key], torch.Tensor) else batch[key]

        model_outputs = model.generate(**batch)
        output_text = model.tokenizer.batch_decode(model_outputs, add_special_tokens=False, skip_special_tokens=True)

        for key, text in zip(batch["keys"], output_text):
            if key not in decode_log_data:
                decode_log_data[key] = {}
            decode_log_data[key]["CAP"] = text.replace("\n", " ")

    # Save updated decode_log
    with open(decode_log_path, "w") as json_file:
        json.dump(decode_log_data, json_file, indent=4)
    logger.info(f"Decode log updated and saved to {decode_log_path}")

if __name__ == "__main__":
    main_hydra()
