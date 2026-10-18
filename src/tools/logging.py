import logging
import sys

from .settings import Settings

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEBUG_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
)


def configure_logging(settings: Settings) -> None:
    """Configure root logging once, with timestamps and a stable format.

    DEBUG mode forces the DEBUG level and a more verbose format; otherwise the
    level comes from LOG_LEVEL (default INFO).
    """
    level = settings.resolved_log_level()
    log_format = DEBUG_LOG_FORMAT if settings.DEBUG else LOG_FORMAT

    logging.basicConfig(
        level=level,
        format=log_format,
        stream=sys.stdout,
        force=True,
    )

    logger = logging.getLogger(__name__)
    logger.debug(
        "Logging configured (log_level=%s, debug=%s)",
        logging.getLevelName(level),
        settings.DEBUG,
    )
