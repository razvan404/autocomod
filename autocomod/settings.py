from typing import ClassVar
import sys
import json

from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
)


class CliArgsSource(EnvSettingsSource):
    def __init__(self, settings_cls: type[BaseSettings], prefix: str | None = None):
        super().__init__(
            settings_cls, env_prefix=prefix or "", env_nested_delimiter="__"
        )
        self._prefix = prefix or ""
        self.env_vars = self._load_args()

    def _load_args(self):
        args = sys.argv[1:]
        env_vars = {}

        def is_value(i: int):
            return i < len(args) and not args[i].startswith("--")

        idx = 0
        while idx < len(args):
            if not args[idx].startswith(f"--{self._prefix}"):
                continue

            current_arg = args[idx][2 + len(self._prefix) :]
            if "=" in current_arg:
                key, value = current_arg.split("=", 1)
            elif is_value(idx + 1):
                key = current_arg
                value = args[idx + 1]
                idx += 1
            else:
                key = args[idx]
                value = True

            if is_value(idx + 1):
                value = [value]
                while is_value(idx + 1):
                    value.append(args[idx + 1])
                    idx += 1
                value = json.dumps(value)

            env_vars[key] = value
            idx += 1

        return env_vars


class Settings(BaseSettings):
    _cli_prefix: ClassVar[str] = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources = [
            init_settings,
            env_settings,
            CliArgsSource(settings_cls, cls._cli_prefix),
        ]
        return tuple(sources)
