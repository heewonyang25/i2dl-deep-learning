from .base_tests import UnitTest, CompositeTest, MethodTest, test_results_to_score
import numpy as np
import math


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


class OptimizerStepTest(UnitTest):
    def __init__(self, Optimizer):
        starting_W = np.copy(Optimizer.model.W)
        sample_grad = np.array([1, 2, 3]).reshape(3, 1)
        Optimizer.step(sample_grad)
        self.truth = starting_W - Optimizer.lr * sample_grad
        self.value = Optimizer.model.W

    def test(self):

        return (rel_error(self.truth, self.value) < 1e-6)

    def define_failure_message(self):
        return "Optimizer Step incorrect.\nExpected: " + \
            str(self.truth) + "\nEvaluated: " + str(self.value)


class OptimizerTest(CompositeTest):
    def define_tests(self, Optimizer):
        return [
            OptimizerStepTest(Optimizer)
        ]

    def define_success_message(self):
        return "Congratulations you have passed all the unit tests!!!"

    def define_failure_message(self):
        return "Test cases are still failing!"


def test_optimizer(Optimizer):
    """Test the Optimizer"""
    test = OptimizerTest(Optimizer)
    return test_results_to_score(test())
