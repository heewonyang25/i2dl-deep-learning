from .base_tests import UnitTest, CompositeTest, MethodTest, test_results_to_score
import numpy as np
import math


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


class Sigmoid_Of_Zero(UnitTest):
    """Test whether Sigmoid of 0 is correct"""

    def __init__(self, Classifier):
        self.value = Classifier.sigmoid(np.float(0))
        self.truth = 0.5

    def test(self):
        return (rel_error(self.truth, self.value) < 1e-6)

    def define_failure_message(self):
        return "Classifier Sigmoid incorrect.\nExpected: " + \
            str(self.truth) + "\nEvaluated: " + str(self.value)


class Sigmoid_Of_Zero_Array(UnitTest):
    """Test whether Sigmoid of a numpy array [0, 0, 0, 0, 0] is correct"""

    def __init__(self, Classifier):
        self.value = Classifier.sigmoid(np.asarray([0, 0, 0, 0, 0])).sum()
        self.truth = 2.5

    def test(self):
        return (rel_error(self.truth, self.value) < 1e-6)

    def define_failure_message(self):
        return "Classifier Sigmoid incorrect.\nExpected: " + \
            str(self.truth) + "\nEvaluated: " + str(self.value)


class Sigmoid_Of_100(UnitTest):
    """Test whether Sigmoid of 100 is correct"""

    def __init__(self, Classifier):
        self.value = Classifier.sigmoid(np.float(100))
        self.truth = 1.0

    def test(self):
        return (rel_error(self.truth, self.value) < 1e-6)

    def define_failure_message(self):
        return "Classifier Sigmoid incorrect.\nExpected: " + \
            str(self.truth) + "\nEvaluated: " + str(self.value)


class Sigmoid_Of_Array_of_100(UnitTest):
    """Test whether Sigmoid of [100, 100, 100, 100, 100] is correct"""

    def __init__(self, Classifier):
        self.value = Classifier.sigmoid(
            np.asarray([100, 100, 100, 100, 100])).sum()
        self.truth = 5.0

    def test(self):
        return (rel_error(self.truth, self.value) < 1e-6)

    def define_failure_message(self):
        return "Classifier Sigmoid incorrect.\nExpected: " + \
            str(self.truth) + "\nEvaluated: " + str(self.value)


class ClassifierForwardTest(UnitTest):
    def __init__(self, Classifier):
        Classifier.initialize_weights()
        sample_x = np.array([1, 2,
                             3, 4]).reshape(2, 2)
        self.value = Classifier.forward(sample_x)
        self.truth = Classifier.sigmoid(np.concatenate(
            (sample_x, np.ones((2, 1))), axis=1).dot(Classifier.W))

    def test(self):

        return (rel_error(self.truth, self.value) < 1e-6)

    def define_failure_message(self):
        return "Classifier forward incorrect.\nExpected: " + \
            str(self.truth) + "\nEvaluated: " + str(self.value)


class ClassifierSigmoidTest(UnitTest):
    def __init__(self, Classifier):
        sample_x = np.array([1, 2,
                             3, 4]).reshape(2, 2)
        self.value = Classifier.sigmoid(sample_x)
        self.truth = 1 / (1 + np.exp(-sample_x))

    def test(self):
        return (rel_error(self.truth, self.value) < 1e-6)

    def define_failure_message(self):
        return "Sigmoid incorrect. \nExpected: " + \
            str(self.truth) + " \nEvaluated: " + str(self.value)


class ClassifierBackwardTest(UnitTest):
    def __init__(self, Classifier):
        Classifier.initialize_weights()
        sample_x = np.array([1, 2,
                             3, 4]).reshape(2, 2)
        sample_y = Classifier.forward(sample_x)
        self.value = Classifier.backward(sample_y)
        self.truth = np.concatenate(
            (sample_x, np.ones((2, 1))), axis=1) * sample_y * (1 - sample_y)

    def test(self):
        return (rel_error(self.truth, self.value) < 1e-6)

    def define_failure_message(self):
        return "Classifier Backward incorrect. \nExpected: " + \
            str(self.truth) + " \nEvaluated: " + str(self.value)


class SigmoidMethodTest(MethodTest):
    def define_tests(self, Classifier):
        return [Sigmoid_Of_Zero(Classifier),
                Sigmoid_Of_Zero_Array(Classifier),
                Sigmoid_Of_100(Classifier),
                Sigmoid_Of_Array_of_100(Classifier)
                ]

    def define_method_name(self):
        return "sigmoid"


class ForwardMethodTest(MethodTest):
    def define_tests(self, Classifier):
        return [ClassifierForwardTest(Classifier)
                ]

    def define_method_name(self):
        return "forward"


class BackwardMethodTest(MethodTest):
    def define_tests(self, Classifier):
        return [ClassifierBackwardTest(Classifier)
                ]

    def define_method_name(self):
        return "backward"


class ClassifierTest(CompositeTest):
    def define_tests(self, Classifier):
        return [SigmoidMethodTest(Classifier),
                ForwardMethodTest(Classifier),
                BackwardMethodTest(Classifier)
                ]

    def define_success_message(self):
        return "Congratulations you have passed all the unit tests!!!"

    def define_failure_message(self):
        return "Test cases are still failing!"


def test_classifier(Classifier):
    """Test the Classifier"""
    test = ClassifierTest(Classifier)
    return test_results_to_score(test())
