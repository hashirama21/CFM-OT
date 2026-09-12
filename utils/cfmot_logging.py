import logging
from typing import Dict


class CFMOTLogger(logging.Logger):
    """Project logger with a single stream handler."""

    def __init__(self, name: str = "cfmot", level: int = logging.INFO):
        super().__init__(name=name, level=level)
        if not self.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.addHandler(handler)
        self.propagate = False


_LOGGERS: Dict[str, CFMOTLogger] = {}


def get_logger(name: str = "cfmot", level: int = logging.INFO) -> CFMOTLogger:
    """Return a cached CFMOTLogger instance by name."""
    logger = _LOGGERS.get(name)
    if logger is None:
        logger = CFMOTLogger(name=name, level=level)
        _LOGGERS[name] = logger
    else:
        logger.setLevel(level)
    return logger
