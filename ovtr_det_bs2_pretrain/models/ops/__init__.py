try:
    from . import _C as MSDA

    HAS_MSDA_EXT = True
    MSDA_IMPORT_ERROR = None
except ImportError as exc:
    MSDA = None
    HAS_MSDA_EXT = False
    MSDA_IMPORT_ERROR = exc


__all__ = ["MSDA", "HAS_MSDA_EXT", "MSDA_IMPORT_ERROR"]
