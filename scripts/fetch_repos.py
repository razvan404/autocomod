import os
import sys

from git import Repo, CommandError

from autocomod.settings import Settings
from autocomod.logger import logger


class FetcherSettings(Settings):
    repos_dir: str = "repos/"
    repos_urls: str | list[str] | None = None
    repos_urls_file: str = "files/default_repo_list.txt"


def fetch_repos(settings: FetcherSettings) -> None:
    if settings.repos_urls is None:
        logger.info(f"Using repos located at `{settings.repos_urls_file}`.")
        with open(settings.repos_urls_file, "r") as f:
            repos_urls = [line.strip() for line in f.readlines()]
    else:
        logger.info("Using repos mentioned through the `--repos_urls` argument.")
        repos_urls = settings.repos_urls
        if isinstance(repos_urls, str):
            repos_urls = [repos_urls]

    os.makedirs(settings.repos_dir, exist_ok=True)

    for repo_url in repos_urls:
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        dest = os.path.join(settings.repos_dir, repo_name)
        if os.path.exists(dest):
            logger.info(f"{repo_url} already exists in `{dest}`, skipping.")
            continue

        logger.info(f"Fetching {repo_url} in `{dest}`.")
        try:
            Repo.clone_from(repo_url, dest, depth=1)
            logger.info(f"Finished fetching {repo_url} in `{dest}`.")
        except CommandError as exc:
            logger.error(f"Failed fetching {repo_url}: {exc}")


if __name__ == "__main__":
    fetch_repos(settings=FetcherSettings())
