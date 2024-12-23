from typing import Callable
import sys
import datetime as dt


__all__ = [
    'obj_hooker', 'disk_reconnect_watir', 'timer'
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


def timer(func: Callable):
    def _wrapper(*args, **kwargs):
        start = dt.datetime.now()
        result = func(*args, **kwargs)
        end = dt.datetime.now()
        print(f'{func.__name__} took {(end - start).total_seconds()}')
        return result
    return _wrapper

def disk_reconnect_watir(func: Callable):
    import datetime as dt
    def _wrapper(*args, **kwargs):
        gargs = kwargs.get('gargs', kwargs.get('args', kwargs.get('arg', kwargs.get('garg'))))

        t0 = tn = dt.datetime.now()
        while (tn - t0).seconds < getattr(gargs, 'timeout', 5):
            try:
                _result = func(*args, **kwargs)
                return _result
            except FileNotFoundError as e:
                print(f'Retrying {func.__name__}')
            tn = dt.datetime.now()
        raise e
    return _wrapper