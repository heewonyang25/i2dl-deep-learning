"""Tests for DataLoader in data/dataloader.py"""

import numpy as np
import copy

from .len_tests import LenTest, ConditionedLenTest
from .base_tests import UnitTest, MethodTest, ConditionedMethodTest, ClassTest, test_results_to_score


def get_values_flat(iterable):
    """get all values from a DataLoader/Dataset as a flat list"""
    data = []
    for batch in iterable:
        for value in batch.values():
            if isinstance(value, (list, np.ndarray)):
                for val in value:
                    data.append(val)
            else:
                data.append(value)
    return data


class IterTestIterable(UnitTest):
    """Test whether __iter()__ is iterable"""

    def __init__(self, iterable):
        self.iterable = iterable

    def test(self):
        for _ in self.iterable:
            pass
        return True

    def define_exception_message(self, exception):
        return "Object is not iterable."


class IterTestItemType(UnitTest):
    """Test whether __iter()__ returns correct item type"""

    def __init__(self, iterable, item_type):
        self.iterable = iterable
        self.item_type = item_type
        self.wrong_type = None

    def test(self):
        for item in self.iterable:
            if not isinstance(item, self.item_type):
                self.wrong_type = type(item)
                return False
        return True

    def define_failure_message(self):
        return "Expected items to be of type %s, got %s instead" \
               % (self.item_type, str(type(self.wrong_type)))


class IterTestBatchSize(UnitTest):
    """Test whether __iter__() of DataLoader uses correct batch_size"""

    def __init__(self, dataloader, batch_size):
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.wrong_batch_size = -1

    def test(self):
        if self.batch_size is None:
            return True
        for batch in self.dataloader:
            for _, value in batch.items():
                if len(value) != self.batch_size:
                    self.wrong_batch_size = len(value)
                    return False
        return True

    def define_failure_message(self):
        return "Wrong batch size (expected %d, got %d)." \
               % (self.batch_size, self.wrong_batch_size)


class IterTestNumBatches(UnitTest):
    """Test whether __iter__() of DataLoader loads correct number of batches"""

    def __init__(self, dataloader, num_batches):
        self.dataloader = dataloader
        self.num_batches = num_batches
        self.num_batches_iter = -1

    def test(self):
        self.num_batches_iter = 0
        for _ in self.dataloader:
            self.num_batches_iter += 1
        return self.num_batches_iter == self.num_batches

    def define_failure_message(self):
        return "Wrong number of batches (expected %d, got %d)." \
               % (self.num_batches, self.num_batches_iter)


class IterTestValuesUnique(UnitTest):
    """Test whether __iter__() of DataLoader loads all values only once"""

    def __init__(self, dataloader):
        self.dataloader = dataloader

    def test(self):
        data = get_values_flat(self.dataloader)
        return len(data) == len(set(data))

    def define_failure_message(self):
        return "Values loaded are not unique."


class IterTestValueRange(UnitTest):
    """Test whether __iter__() of DataLoader loads correct value range"""

    def __init__(self, dataloader, min_, max_):
        self.dataloader = dataloader
        self.min = min_
        self.max = max_
        self.min_iter = -1
        self.max_iter = -1

    def test(self):
        if self.min is None or self.max is None:
            return True
        data = get_values_flat(self.dataloader)
        self.min_iter = min(data)
        self.max_iter = max(data)
        return self.min_iter >= self.min and self.max_iter <= self.max

    def define_failure_message(self):
        return "Expected lowest and highest value to be %d and %d, " \
               "but got minimum value %d and maximum value %d." \
               % (self.min, self.max, self.min_iter, self.max_iter)


class IterTestShuffled(UnitTest):
    """Test whether __iter__() of DataLoader shuffles the data"""

    def __init__(self, dataloader, shuffle):
        self.dataloader = dataloader
        self.shuffle = shuffle

    def test(self):
        if not self.shuffle:
            return True
        data = get_values_flat(self.dataloader)
        return data != sorted(data)

    def define_failure_message(self):
        return "Data loaded seems to be not shuffled."


class IterTestNonDeterministic(UnitTest):
    """Test whether __iter__() of DataLoader shuffles the data"""

    def __init__(self, dataloader, shuffle):
        self.dataloader = dataloader
        self.shuffle = shuffle

    def test(self):
        if not self.shuffle:
            return True
        data1 = get_values_flat(self.dataloader)
        data2 = get_values_flat(self.dataloader)
        return data1 != data2

    def define_failure_message(self):
        return "Loading seems to be deterministic, even though shuffle=True."


class IterTest(MethodTest):
    """Test __iter__() method of DataLoader"""

    def define_tests(
            self, dataloader, batch_size, len_, min_val, max_val, shuffle
    ):
        return [
            IterTestIterable(dataloader),
            IterTestItemType(dataloader, dict),
            IterTestBatchSize(dataloader, batch_size),
            IterTestNumBatches(dataloader, len_),
            IterTestValuesUnique(dataloader),
            IterTestValueRange(dataloader, min_val, max_val),
            IterTestShuffled(dataloader, shuffle),
            IterTestNonDeterministic(dataloader, shuffle)
        ]

    def define_method_name(self):
        return "__iter__"
   

class ConditionedIterTest(ConditionedMethodTest):
    """Test __iter__() method of DataLoader"""

    def define_tests(
            self, dataloader, batch_size, len_, min_val, max_val, shuffle
    ):
        return [
            IterTestIterable(dataloader),
            IterTestItemType(dataloader, dict),
            IterTestBatchSize(dataloader, batch_size),
            IterTestNumBatches(dataloader, len_),
            IterTestValuesUnique(dataloader),
            IterTestValueRange(dataloader, min_val, max_val),
            IterTestShuffled(dataloader, shuffle),
            IterTestNonDeterministic(dataloader, shuffle)
        ]

    def define_method_name(self):
        return "__iter__"



class DataLoaderTest(ClassTest):
    """Test DataLoader class"""

    def define_tests(
            self, dataloader, batch_size, len_, min_val, max_val, shuffle
    ):
        return [
            LenTest(dataloader, len_),
            IterTest(dataloader, batch_size, len_, min_val, max_val, shuffle),
        ]

    def define_class_name(self):
        return "DataLoader"


def test_dataloader(
        dataset,
        dataloader,
        batch_size=1,
        shuffle=False,
        drop_last=False
):
    """Test DataLoader class"""
    if drop_last:
        test = DataLoaderTest(
            dataloader,
            batch_size=batch_size,
            len_=len(dataset) // batch_size,
            min_val=None,
            max_val=None,
            shuffle=shuffle,
        )
    else:
        test = DataLoaderTest(
            dataloader,
            batch_size=None,
            len_=int(np.ceil(len(dataset) / batch_size)),
            min_val=min(get_values_flat(dataset)),
            max_val=max(get_values_flat(dataset)),
            shuffle=shuffle,
        )
    return test_results_to_score(test())


# Implementations for  testing of __len__() method


class DataloaderLenTest(MethodTest):
    """Test __len__() method of DataLoader for both drop_last modi"""

    def define_tests(
            self, dataloader
    ):
        dataset = dataloader.dataset
        batch_size = dataloader.batch_size
        
        drop_last_dataloader = copy.copy(dataloader)
        drop_last_dataloader.drop_last = True
        
        all_dataloader = copy.copy(dataloader)
        all_dataloader.drop_last = False
        
        return [
            ConditionedLenTest(
                'drop_last=True',    
                drop_last_dataloader,
                len(dataset) // batch_size
               ),
            ConditionedLenTest(
                'drop_last=False',
                all_dataloader,
                int(np.ceil(len(dataset) / batch_size))
               )
        ]

    def define_method_name(self):
        return "__len__"

    
def test_dataloader_len(dataloader):
    test = DataloaderLenTest(dataloader)
    return test_results_to_score(test())


# Implementations for  testing of __iter__() method


class DataloaderIterTest(MethodTest):
    """Test __len__() method of DataLoader for both drop_last modi"""
    
    def define_tests(
            self, dataloader
    ):
        batch_size = dataloader.batch_size
        shuffle = dataloader.shuffle
        
        drop_last_dataloader = copy.copy(dataloader)
        drop_last_dataloader.drop_last = True
        dataset = drop_last_dataloader.dataset
        min_val_drop=min(get_values_flat(dataset))
        max_val_drop=max(get_values_flat(dataset))
        len_drop = len(dataset) // batch_size
        batch_size_drop = batch_size
        
        all_dataloader = copy.copy(dataloader)
        all_dataloader.drop_last = False
        dataset = all_dataloader.dataset
        min_val_all=min(get_values_flat(dataset))
        max_val_all=max(get_values_flat(dataset))
        len_all = int(np.ceil(len(dataset) / batch_size))
        batch_size_all = None
        
        return [
            ConditionedIterTest(
                'drop_last=True',    
                drop_last_dataloader,
                len_=len_drop,
                batch_size=batch_size_drop,
                shuffle=shuffle,
                min_val=min_val_drop,
                max_val=max_val_drop
               ),
            ConditionedIterTest(
                'drop_last=False',
                all_dataloader,
                len_=len_all,
                batch_size=batch_size_all,
                shuffle=shuffle,
                min_val=min_val_all,
                max_val=max_val_all
               )
        ]

    def define_method_name(self):
        return "__iter__"
    

def test_dataloader_iter(
        dataloader
):    
    test = DataloaderIterTest(
        dataloader
    )
    return test_results_to_score(test())



# Implementations for  testing of __iter__() method seperately for item type,batch size,
# values and shuffle

class IterItemTest(MethodTest):
    """Test __iter__() method of DataLoader"""

    def define_tests(
            self, dataloader, batch_size, len_, min_val, max_val, shuffle
    ):
        return [
            IterTestIterable(dataloader),
            IterTestItemType(dataloader, dict),
        ]

    def define_method_name(self):
        return "__iter__"


class IterBatchTest(MethodTest):
    """Test __iter__() method of DataLoader"""

    def define_tests(
            self, dataloader, batch_size, len_, min_val, max_val, shuffle
    ):
        return [
            IterTestBatchSize(dataloader, batch_size),
            IterTestNumBatches(dataloader, len_),

        ]

    def define_method_name(self):
        return "__iter__"


class IterValueTest(MethodTest):
    """Test __iter__() method of DataLoader"""

    def define_tests(
            self, dataloader, batch_size, len_, min_val, max_val, shuffle
    ):
        return [
            IterTestValuesUnique(dataloader),
            IterTestValueRange(dataloader, min_val, max_val),

        ]

    def define_method_name(self):
        return "__iter__"


class IterShuffleTest(MethodTest):
    """Test __iter__() method of DataLoader"""

    def define_tests(
            self, dataloader, batch_size, len_, min_val, max_val, shuffle
    ):
        return [
            IterTestShuffled(dataloader, shuffle),
            IterTestNonDeterministic(dataloader, shuffle)
        ]

    def define_method_name(self):
        return "__iter__"


def test_iter_shuffle(
        dataset,
        dataloader,
        batch_size=1,
        shuffle=False,
        drop_last=False
):
    """Test DataLoader class"""
    if drop_last:
        test = IterShuffleTest(
            dataloader,
            batch_size=batch_size,
            len_=len(dataset) // batch_size,
            min_val=None,
            max_val=None,
            shuffle=shuffle,
        )
    else:
        test = IterShuffleTest(
            dataloader,
            batch_size=None,
            len_=int(np.ceil(len(dataset) / batch_size)),
            min_val=min(get_values_flat(dataset)),
            max_val=max(get_values_flat(dataset)),
            shuffle=shuffle,
        )
    return test_results_to_score(test())


def test_iter_value(
        dataset,
        dataloader,
        batch_size=1,
        shuffle=False,
        drop_last=False
):
    """Test DataLoader class"""
    if drop_last:
        test = IterValueTest(
            dataloader,
            batch_size=batch_size,
            len_=len(dataset) // batch_size,
            min_val=None,
            max_val=None,
            shuffle=shuffle,
        )
    else:
        test = IterValueTest(
            dataloader,
            batch_size=None,
            len_=int(np.ceil(len(dataset) / batch_size)),
            min_val=min(get_values_flat(dataset)),
            max_val=max(get_values_flat(dataset)),
            shuffle=shuffle,
        )
    return test_results_to_score(test())


def test_iter_batch(
        dataset,
        dataloader,
        batch_size=1,
        shuffle=False,
        drop_last=False
):
    """Test DataLoader class"""
    if drop_last:
        test = IterBatchTest(
            dataloader,
            batch_size=batch_size,
            len_=len(dataset) // batch_size,
            min_val=None,
            max_val=None,
            shuffle=shuffle,
        )
    else:
        test = IterValueTest(
            dataloader,
            batch_size=None,
            len_=int(np.ceil(len(dataset) / batch_size)),
            min_val=min(get_values_flat(dataset)),
            max_val=max(get_values_flat(dataset)),
            shuffle=shuffle,
        )
    return test_results_to_score(test())


def test_iter_item(
        dataset,
        dataloader,
        batch_size=1,
        shuffle=False,
        drop_last=False
):
    """Test DataLoader class"""
    if drop_last:
        test = IterItemTest(
            dataloader,
            batch_size=batch_size,
            len_=len(dataset) // batch_size,
            min_val=None,
            max_val=None,
            shuffle=shuffle,
        )
    else:
        test = IterItemTest(
            dataloader,
            batch_size=None,
            len_=int(np.ceil(len(dataset) / batch_size)),
            min_val=min(get_values_flat(dataset)),
            max_val=max(get_values_flat(dataset)),
            shuffle=shuffle,
        )
    return test_results_to_score(test())
