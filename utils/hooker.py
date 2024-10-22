from typing import Callable
import sys


__all__ = [
    'obj_hooker', 'disk_reconnect_watir'
]


def obj_hooker(func: Callable):
    def _wrapper(obj, *args, **kwargs):
        try:
            _result = func(obj, *args, **kwargs)
            return _result
        except Exception as e:
            except_info = str(e)
            debug_info = obj.debug_card()
            tb = sys.exc_info()[2]
            raise type(e)(f'{except_info}\n{debug_info}').with_traceback(tb)
    return _wrapper


def disk_reconnect_watir(func: Callable):
    import datetime as dt
    def _wrapper(*args, **kwargs):
        gargs = kwargs.get('gargs', kwargs.get('args', kwargs.get('arg', kwargs.get('garg'))))

        t0 = tn = dt.datetime.now()
        while (tn - t0).seconds < gargs.timeout:
            try:
                _result = func(*args, **kwargs)
                return _result
            except FileNotFoundError as e:
                print(f'Retrying {func.__name__}')
            tn = dt.datetime.now()
        raise e
    return _wrapper