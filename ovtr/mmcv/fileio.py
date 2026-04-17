class BaseStorageBackend:
    def get(self, filepath):
        raise NotImplementedError

    def get_text(self, filepath):
        raise NotImplementedError


class HardDiskBackend(BaseStorageBackend):
    def get(self, filepath):
        with open(filepath, "rb") as handle:
            return handle.read()

    def get_text(self, filepath):
        with open(filepath, "r", encoding="utf-8") as handle:
            return handle.read()


class FileClient:
    _backends = {"disk": HardDiskBackend}

    @classmethod
    def register_backend(cls, name, force=False):
        def decorator(backend_cls):
            if not force and name in cls._backends:
                raise KeyError(f"backend {name} is already registered")
            cls._backends[name] = backend_cls
            return backend_cls

        return decorator

    def __init__(self, backend="disk", **kwargs):
        if backend not in self._backends:
            raise KeyError(f"backend {backend} is not registered")
        self.backend = self._backends[backend](**kwargs)

    def get(self, filepath):
        return self.backend.get(filepath)

    def get_text(self, filepath):
        return self.backend.get_text(filepath)
