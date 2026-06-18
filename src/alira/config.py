import os


class _Config:
    def __getitem__(self, key: str) -> str | None:
        return os.environ.get(key)


CONFIG: _Config = _Config()
