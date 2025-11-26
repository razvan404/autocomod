import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from autocomod.parsers import ExtractionParser
from autocomod.logger import logger
from autocomod.settings import Settings


class ExtractionsProcessorSettings(Settings):
    repos_file: str = "data/repos.csv"
    extractions_dir: str = "data/raw_extractions"
    processed_dir: str = "data/processed_extractions"
    max_workers: int = 12


def process_single_extraction(repo_name: str, exaction_path: Path, output_path: Path):
    with open(exaction_path, "r") as f:
        extraction_data = json.load(f)

    logger.info(f"Processing extraction for repo `{repo_name}`")
    parser = ExtractionParser(extraction_data)
    processed_extraction_data = parser.compute_processed_data()
    with open(output_path, "w") as f:
        json.dump(processed_extraction_data, f, indent=2)
    logger.info(
        f"Saved processed extraction data for repo `{repo_name}` in `{exaction_path}`"
    )


def process_extractions(settings: ExtractionsProcessorSettings):
    repos = pd.read_csv(settings.repos_file)
    extractions_dir = Path(settings.extractions_dir)
    processed_dir = Path(settings.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
        futures = []
        for _, repo_data in repos.iterrows():
            repo_name = repo_data.id
            extraction_path = extractions_dir / f"{repo_name}.json"

            if not extraction_path.exists():
                logger.warning(
                    f"Extractions for repo {repo_name} not found in `{extraction_path}`, skipping"
                )
                continue

            output_path = processed_dir / f"{repo_name}.json"
            if output_path.exists():
                logger.warning(
                    f"Processed extraction file {output_path} already exists, skipping"
                )
                continue

            future = executor.submit(
                process_single_extraction, repo_name, extraction_path, output_path
            )
            futures.append(future)

        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    process_extractions(settings=ExtractionsProcessorSettings())
