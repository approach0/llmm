import os
import errno
import signal
from functools import wraps


def timeout(seconds=100, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)
        @wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator


@timeout(seconds=2)
def test():
    import time
    time.sleep(4)


if __name__ == '__main__':
    try:
        test()
    except TimeoutError as e:
        print(str(e))
