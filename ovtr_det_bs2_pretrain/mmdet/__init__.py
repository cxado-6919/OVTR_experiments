from pathlib import Path


_SHARED_DIR = Path(__file__).resolve().parents[2] / "ovtr" / "mmdet"
__file__ = str(_SHARED_DIR / "__init__.py")
__path__ = [str(_SHARED_DIR)]
if "__spec__" in globals() and __spec__ is not None:
    __spec__.submodule_search_locations = __path__

with open(__file__, "r", encoding="utf-8") as handle:
    exec(compile(handle.read(), __file__, "exec"), globals(), globals())
