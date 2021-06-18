import traceback


def ignore_exception(f):

    def apply_func(*args, **kwargs):
        try:
            result = f(*args, **kwargs)
            return result
        except Exception:
            print(f'Catched exception in {f}:')
            traceback.print_exc()
            return None
    return apply_func