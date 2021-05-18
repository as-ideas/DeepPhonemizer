import unittest

from dp.training.decorators import ignore_exception


@ignore_exception
def test_function() -> None:
    raise ValueError('Exception should be ignored.')


class TestDecorators(unittest.TestCase):

    def test_ignore_exception(self):
        test_function()

