from .base_tests import UnitTest, CompositeTest, MethodTest, test_results_to_score
import numpy as np
import math


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


class SolverStepTest(UnitTest):
    """Test whether Solver._step() updates the model parameter correctly"""

    def __init__(self, Solver):
        Solver._step()
        self.truth = [[0.11574258], [0.0832162]]
        self.value = Solver.model.W
        pass

    def test(self):

        return (rel_error(self.truth, self.value) < 1e-6)

    def define_failure_message(self):
        return "Solver Step incorrect.\nExpected: " + \
            str(self.truth) + "\nEvaluated: " + str(self.value)


class SolverTest(MethodTest):
    def define_tests(self, Solver):
        return [
            SolverStepTest(Solver)
        ]

    def define_success_message(self):
        return "Congratulations you have passed all the unit tests!!!"

    def define_failure_message(self):
        return "Test cases are still failing!"

    def define_method_name(self):
        return "_step"


def test_solver(Solver):
    """Test the Solver"""
    test = SolverTest(Solver)
    return test_results_to_score(test())
