import logging

from colorlog import ColoredFormatter


def _create_logger(name: str, level: int = logging.DEBUG):
    formatter = ColoredFormatter(
        "%(asctime)s.%(msecs)03d - %(log_color)s%(levelname)s%(reset)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(level)

    return logger


logger = _create_logger("autocomod")
