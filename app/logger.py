import sys

from loguru import logger


def setup_logging(json_output: bool = True) -> None:
    """Configure loguru for structured JSON logging."""
    logger.remove()

    if json_output:
        logger.add(
            sys.stderr,
            format="{message}",
            serialize=True,
            level="INFO",
        )
    else:
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="DEBUG",
        )
