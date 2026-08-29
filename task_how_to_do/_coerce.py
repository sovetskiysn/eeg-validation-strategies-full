"""Приведение YAML-структур к типам, которые ждёт mne-bids-pipeline."""
import typing
from collections.abc import Sequence as ABCSequence

from mne_bids_pipeline import _config as mbp_config


class _Mismatch(TypeError):
    pass


def coerce(value, annot):
    """list<->tuple по аннотации. Неизвестные типы отдаём как есть."""
    origin = typing.get_origin(annot)

    if origin is typing.Annotated:          # снимаем валидаторы pydantic
        return coerce(value, typing.get_args(annot)[0])

    if origin in (typing.Union, __import__("types").UnionType):
        for member in typing.get_args(annot):
            try:
                return coerce(value, member)
            except _Mismatch:
                continue
        raise _Mismatch(annot)

    if annot is type(None):
        if value is None:
            return None
        raise _Mismatch(annot)

    if origin is typing.Literal:
        if value in typing.get_args(annot):
            return value
        raise _Mismatch(annot)

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise _Mismatch(annot)
        args = typing.get_args(annot)
        if len(args) == 2 and args[1] is Ellipsis:      # tuple[X, ...]
            return tuple(coerce(v, args[0]) for v in value)
        if len(args) != len(value):
            raise _Mismatch(annot)
        return tuple(coerce(v, a) for v, a in zip(value, args))

    if origin in (list, ABCSequence):
        if isinstance(value, str) or not isinstance(value, (list, tuple)):
            raise _Mismatch(annot)
        (arg,) = typing.get_args(annot) or (typing.Any,)
        return [coerce(v, arg) for v in value]

    if origin is dict:
        if not isinstance(value, dict):
            raise _Mismatch(annot)
        _, vt = typing.get_args(annot) or (typing.Any, typing.Any)
        return {k: coerce(v, vt) for k, v in value.items()}

    if isinstance(annot, type):             # str / int / float / bool
        if annot is float and isinstance(value, int) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, annot) and not (annot is not bool and isinstance(value, bool)):
            return value
        raise _Mismatch(annot)

    return value                            # ArrayLike и прочая экзотика


def coerce_all(params: dict) -> dict:
    ann = mbp_config.__annotations__
    return {k: coerce(v, ann[k]) if k in ann else v for k, v in params.items()}
