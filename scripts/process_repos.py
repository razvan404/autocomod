import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from autocomod.parsers import RepoParser
from autocomod.logger import logger
from autocomod.settings import Settings


class ReposProcessorSettings(Settings):
    repos_dir: str = "repos/"
    extractions_dir: str = "data/raw_extractions"
    repos_file: str = "data/repos.csv"
    repo_names: str | list[str] | None = None
    max_workers: int = 1


def process_single_repo(
    repo_name: str, repo_src: str, repos_dir: str, graph_data_file: Path
) -> None:
    repo_path = Path(repos_dir) / repo_name / repo_src

    if not repo_path.exists():
        logger.warning(f"Repo `{repo_name}` doesn't exist in `{repo_path}`.")
        return

    logger.info(f"Preprocessing repo `{repo_name}`.")

    try:
        parser = RepoParser(repo_path)
        nodes_data, edges_data = parser.compute_graph_data()
    except Exception as exc:
        logger.error(f"Failed to parse `{repo_name}`: {exc}")
        return

    with open(graph_data_file, "w") as f:
        json.dump({"nodes": nodes_data, "edges": edges_data}, f, indent=2)
    logger.info(f"Saved graph data of repo `{repo_name}` in `{graph_data_file}`.")


def process_repos(settings: ReposProcessorSettings):
    repos = pd.read_csv(settings.repos_file)

    if settings.repo_names:
        if isinstance(settings.repo_names, str):
            settings.repo_names = [settings.repo_names]
        repos = repos[repos.id.isin(settings.repo_names)]
    extractions_dir = Path(settings.extractions_dir)
    extractions_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
        futures = []
        for _, repo_data in repos.iterrows():
            repo_name, repo_src = repo_data.id, repo_data.source
            graph_data_file = extractions_dir / f"{repo_name}.json"

            if graph_data_file.exists():
                logger.warning(
                    f"Skipping `{repo_name}`, already processed in `{graph_data_file}`"
                )
                continue

            future = executor.submit(
                process_single_repo,
                repo_name,
                repo_src,
                settings.repos_dir,
                graph_data_file,
            )
            futures.append(future)

        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    process_repos(settings=ReposProcessorSettings())
