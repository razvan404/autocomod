import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch
import pandas as pd

from autocomod.parsers import PygParser
from autocomod.logger import logger
from autocomod.settings import Settings


class PygProcessorSettings(Settings):
    repos_file: str = "data/repos.csv"
    processed_dir: str = "data/processed_extractions"
    pyg_dir: str = "data/pyg_graphs"
    max_workers: int = 12


def process_single_pyg(repo_name: str, input_path: Path, output_path: Path) -> None:
    logger.info(f"[{repo_name}] Loading processed extraction JSON...")
    with open(input_path, "r") as f:
        graph_dict: dict[str, Any] = json.load(f)

    logger.info(f"[{repo_name}] Converting graph to PyG Data...")
    data = PygParser(graph_dict).to_pyg_data(repo_name=repo_name)

    logger.info(f"[{repo_name}] Saving PyG Data to `{output_path}`")
    torch.save(data, output_path)


def process_all_pyg(settings: PygProcessorSettings) -> None:
    repos = pd.read_csv(settings.repos_file)
    processed_dir = Path(settings.processed_dir)
    pyg_dir = Path(settings.pyg_dir)

    pyg_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
        futures = []

        for _, repo_data in repos.iterrows():
            repo_name = repo_data.id
            input_path = processed_dir / f"{repo_name}.json"

            if not input_path.exists():
                logger.warning(
                    f"[{repo_name}] Processed extraction file not found at `{input_path}`, skipping"
                )
                continue

            output_path = pyg_dir / f"{repo_name}.pt"
            if output_path.exists():
                logger.warning(
                    f"[{repo_name}] PyG Data file `{output_path}` already exists, skipping"
                )
                continue

            future = executor.submit(
                process_single_pyg,
                repo_name,
                input_path,
                output_path,
            )
            futures.append(future)

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error during PyG conversion: {e}")


if __name__ == "__main__":
    process_all_pyg(PygProcessorSettings())
