from __future__ import annotations

from typing import Callable

from .base import FailureDetector

_MODELS: dict[str, type[FailureDetector]] = {}


def register_model(name: str) -> Callable:
    def decorator(cls: type[FailureDetector]):
        if name in _MODELS and _MODELS[name] is not cls:
            raise ValueError(f"Duplicate model registration: {name}")
        _MODELS[name] = cls
        cls.model_type = name
        return cls
    return decorator


def list_models() -> list[str]:
    return sorted(_MODELS)


def get_model(name: str, **kwargs) -> FailureDetector:
    if name not in _MODELS:
        valid = ", ".join(list_models())
        raise KeyError(f"Unknown model '{name}'. Valid models: {valid}")
    return _MODELS[name](**kwargs)
