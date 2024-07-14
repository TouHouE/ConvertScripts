from typing import Callable
import sys


__all__ = [
    'obj_hooker'
]


def obj_hooker(func: Callable):
    def _wrapper(obj, *args, **kwargs):
        try:
            _result = func(obj, *args, **kwargs)
        except Exception as e:
            except_info = str(e)
            debug_info = obj.debug_card()
            tb = sys.exc_info()[2]
            raise type(e)(f'{except_info}\n{debug_info}').with_traceback(tb)
    return _wrapper
