import traceback
from time import time


def ignore_exception(f):
    """

    Args:
      f: 

    Returns:

    """
    def apply_func(*args, **kwargs):
        """

        Args:
          *args: 
          **kwargs: 

        Returns:

        """
        try:
            result = f(*args, **kwargs)
            return result
        except Exception:
            print(f'Catched exception in {f}:')
            traceback.print_exc()
            return None
    return apply_func