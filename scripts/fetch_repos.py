import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from git import Repo, CommandError

from autocomod.settings import Settings
from autocomod.logger import logger


class FetcherSettings(Settings):
    repos_dir: str = "repos/"
    repos_file: str = "files/repos.csv"
    max_workers: int = 12


def clone_repo(repo_url: str, dest: str) -> None:
    if os.path.exists(dest):
        logger.info(f"{repo_url} already exists in `{dest}`, skipping.")
        return

    logger.info(f"Fetching {repo_url} in `{dest}`.")
    try:
        Repo.clone_from(repo_url, dest, depth=1)
        logger.info(f"Finished fetching {repo_url} in `{dest}`.")
    except CommandError as exc:
        logger.error(f"Failed fetching {repo_url}: {exc}")
        os.rmdir(dest)


def fetch_repos(settings: FetcherSettings) -> None:
    repos = pd.read_csv(settings.repos_file)
    os.makedirs(settings.repos_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
        futures = []
        for _, repo_data in repos.iterrows():
            repo_name, repo_url = str(repo_data.id), str(repo_data.url)
            dest = os.path.join(settings.repos_dir, repo_name)

            future = executor.submit(clone_repo, repo_url, dest)
            futures.append(future)

        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    fetch_repos(settings=FetcherSettings())
