from abc import ABC, abstractmethod
import random


class UnitTest(ABC):
    """
    Abstract class for a single test
    All subclasses have to overwrite test() and failure_message()
    Then the execution order is the following:
        1. test() method is executed
        2. if test() method returned False or threw an exception,
            print the failure message defined by failure_message()
        3.  return a tuple (tests_failed, total_tests)
    """

    def __call__(self):
        try:
            test_passed = self.test()
            if test_passed:
                print(self.define_success_message())
                # return 0, 1  # 0 tests failed, 1 total test
            print(self.define_failure_message())
            # return 1, 1  # 1 test failed, 1 total test
        except Exception as exception:
            print(self.define_exception_message(exception))
            # return 1, 1  # 1 test failed, 1 total test

    @abstractmethod
    def test(self):
        """Run the test and return True if passed else False"""

    def define_failure_message(self):
        """Define the message that should be printed upon test failure"""
        return "%s failed." % type(self).__name__

    def define_success_message(self):
        """Define the message that should be printed upon test success"""
        return "%s passed." % type(self).__name__

    def define_exception_message(self, exception):
        """
        Define the message that should be printed if an exception occurs
        :param exception: exception that was thrown
        """
        return "%s failed due to exception: %s." \
               % (type(self).__name__, exception)


class DummyTest(UnitTest):
    """Test whether the value is bigger than the threshold"""

    def __init__(self, model):
        self.value = model.forward(random.randint(0, 59))

    def test(self):
        return self.value > 59

    def define_failure_message(self):
        return "The score of your dummy machine is: " + str(self.value)
