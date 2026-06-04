from .email_io import mbox_to_eml, msg_to_eml, pst_to_eml, resolve_email_input
from .logger import Logger, Timer

__all__ = [
    "Logger",
    "Timer",
    "mbox_to_eml",
    "msg_to_eml",
    "pst_to_eml",
    "resolve_email_input",
    "DomainUtils",
]


def __getattr__(name):
    # Lazily expose DomainUtils without eagerly importing the heavy data_utils
    # module (transformers, langdetect, ...) just to use the package.
    if name == "DomainUtils":
        from .data_utils import DomainUtils

        return DomainUtils
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
