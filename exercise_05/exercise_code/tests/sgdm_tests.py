import numpy as np

from .base_tests import UnitTest, CompositeTest, MethodTest, test_results_to_score
from exercise_code.networks.optimizer import *
from .gradient_check import eval_numerical_gradient_array


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


class SGDM_Weight_Test(UnitTest):
    def __init__(self):
        self.sgd_momentum = SGDMomentum._update
        N, D = 4, 5

        w = np.linspace(-0.4, 0.6, num=N*D).reshape(N, D)
        dw = np.linspace(-0.6, 0.4, num=N*D).reshape(N, D)
        v = np.linspace(0.6, 0.9, num=N*D).reshape(N, D)

        config = {'learning_rate': 1e-3, 'velocity': v}
        self.value, _ = self.sgd_momentum(None, w, dw, config=config, lr=1e-3)
        self.truth = np.asarray([
            [0.1406,      0.20738947,  0.27417895,  0.34096842,  0.40775789],
            [0.47454737,  0.54133684,  0.60812632,  0.67491579,  0.74170526],
            [0.80849474,  0.87528421,  0.94207368,  1.00886316,  1.07565263],
            [1.14244211,  1.20923158,  1.27602105,  1.34281053,  1.4096]])

    def test(self):

        self.error = rel_error(self.truth, self.value)

        return self.error < 1e-6

    def define_failure_message(self):
        return "SGD Momentum Weight updates incorrect. Expected: < 1e-6 Evaluated: " + str(self.error)


class SGDM_Velocity_Test(UnitTest):
    def __init__(self):
        self.sgd_momentum = SGDMomentum._update
        N, D = 4, 5

        w = np.linspace(-0.4, 0.6, num=N*D).reshape(N, D)
        dw = np.linspace(-0.6, 0.4, num=N*D).reshape(N, D)
        v = np.linspace(0.6, 0.9, num=N*D).reshape(N, D)

        config = {'learning_rate': 1e-3, 'velocity': v}
        next_w, _ = self.sgd_momentum(None, w, dw, config=config, lr=1e-3)

        self.value = config['velocity']
        self.truth = np.asarray([
            [0.5406,      0.55475789,  0.56891579, 0.58307368,  0.59723158],
            [0.61138947,  0.62554737,  0.63970526,  0.65386316,  0.66802105],
            [0.68217895,  0.69633684,  0.71049474,  0.72465263,  0.73881053],
            [0.75296842,  0.76712632,  0.78128421,  0.79544211,  0.8096]])

    def test(self):

        self.error = rel_error(self.truth, self.value)

        return self.error < 1e-6

    def define_failure_message(self):
        return "SGD Momentum Velocity Values incorrect. Expected: < 1e-6 Evaluated: " + str(self.error)


class SGDMTest(CompositeTest):
    def define_tests(self):
        return [
            SGDM_Weight_Test(),
            SGDM_Velocity_Test()
        ]

    def define_success_message(self):
        return "Congratulations you have passed all the unit tests!!!"

    def define_failure_message(self):
        return "Test cases are still failing!"


class SGDMTestWrapper:
    def __init__(self):
        self.sgdm_tests = SGDMTest()

    def __call__(self, *args, **kwargs):
        return "You secured a score of :" + \
            str(test_results_to_score(self.sgdm_tests()))
