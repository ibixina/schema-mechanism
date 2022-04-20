import unittest
from unittest import TestCase

import numpy as np

from schema_mechanism.util import AccumulatingTrace
from schema_mechanism.util import BoundedSet
from schema_mechanism.util import ReplacingTrace
from schema_mechanism.util import equal_weights
from test_share.test_classes import MockObservable
from test_share.test_classes import MockObserver
from test_share.test_func import common_test_setup


class TestObserverAndObservables(TestCase):
    def setUp(self) -> None:
        common_test_setup()

    def test(self):
        observable = MockObservable()

        # test register
        n_observers = 10
        observers = [MockObserver() for _ in range(n_observers)]

        n_registered = 0
        for obs in observers:
            observable.register(obs)
            n_registered += 1

            self.assertEqual(n_registered, len(observable.observers))
            self.assertTrue(obs in observable.observers)

        # test notify
        observable.notify_all(keyword='kwargs')
        for obs in observers:
            self.assertEqual(1, obs.n_received)
            self.assertEqual('kwargs', obs.last_message['kwargs']['keyword'])

        # test unregister
        removed_obs = observers.pop()
        observable.unregister(removed_obs)

        self.assertEqual(len(observers), len(observable.observers))
        self.assertNotIn(removed_obs, observable.observers)

    def ConcreteObserver(self):
        pass


class TestBoundedSet(TestCase):
    def setUp(self):
        common_test_setup()

        self.legal_values = set(range(10))
        self.illegal_values = {-1, 11, 'illegal', frozenset()}
        self.s = BoundedSet(accepted_values=self.legal_values)

    def test_init(self):
        # test: empty bounded set supported
        s = BoundedSet()
        self.assertSetEqual(set(), s)

        # test: bounded set initialized with non-empty iterable should include all of its values
        s = BoundedSet(self.legal_values, accepted_values=range(10))
        self.assertSetEqual(self.legal_values, s)

        # test: bounded set objects are instances of set
        self.assertIsInstance(s, set)

    def test_is_legal_value(self):
        # test: all legal values should return True
        for v in self.legal_values:
            self.assertTrue(self.s.is_legal_value(v))

        # test: illegal values should return False
        for v in self.illegal_values:
            self.assertFalse(self.s.is_legal_value(v))

    def test_check_values(self):
        # test: legal values should not raise a ValueError
        try:
            self.s.check_values(self.legal_values)
        except ValueError as e:
            self.fail(f'Unexpected ValueError encountered: {str(e)}')

        # test: illegal values should raise a ValueError
        for value in self.illegal_values:
            self.assertRaises(ValueError, lambda: self.s.check_values([value]))

    def test_add(self):
        # test: adding legal values should be included in set and not raise a ValueError
        for v in self.legal_values:
            self.s.add(v)
            self.assertIn(v, self.s)

        # test: illegal values should raise a ValueError on add
        for v in self.illegal_values:
            self.assertRaises(ValueError, lambda: self.s.add(v))

    def test_update(self):
        # test: adding legal values should be included in set and not raise a ValueError
        self.s.update(self.legal_values)
        self.assertSetEqual(self.legal_values, self.s)

        # test: illegal values should raise a ValueError on add
        self.assertRaises(ValueError, lambda: self.s.add(self.illegal_values))


class TestFunctions(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

    def test_equal_weights(self):
        # test: n = 0 should return an empty array
        weights = equal_weights(0)

        self.assertIsInstance(weights, np.ndarray)
        self.assertEqual(0, len(weights))

        # test: n < 0 should raise a ValueError
        self.assertRaises(ValueError, lambda: equal_weights(-1))

        # test: n >= 1 should return numpy ndarray containing n elements that sum close to 1.0
        try:
            for n in range(1, 100):
                weights = equal_weights(n)
                self.assertIsInstance(weights, np.ndarray)
                self.assertEqual(n, len(weights))
                self.assertAlmostEqual(1.0, sum(weights))
        except ValueError as e:
            self.fail(f'Unexpected exception: {str(e)}')


class TestAccumulatingTrace(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

    def test_init(self):
        tr = AccumulatingTrace(decay_rate=0.15, pre_allocated=250)

        self.assertEqual(0.15, tr.decay_rate)
        self.assertEqual(250, tr.n_allocated)

    def test_update_with_empty_set(self):
        tr = AccumulatingTrace()

        # test: adding empty set should not increment length of trace
        tr.update([])

        self.assertEqual(0, len(tr))

    def test_update_add_single(self):
        tr = AccumulatingTrace()

        # test: initial length should be 0
        self.assertEqual(0, len(tr))

        # test: adding a single element should increment length by 1
        tr.update(['1'])
        self.assertEqual(1, len(tr))

        # test: add an element that already exists should not increment the length
        tr.update(['1'])
        self.assertEqual(1, len(tr))

        # test: adding a new element that does not exist should increment length by 1
        tr.update(['2'])
        self.assertEqual(2, len(tr))

    def test_update_add_multiple(self):
        tr = AccumulatingTrace()

        # test: adding multiple elements should increment length by the number of elements
        e1 = [str(i) for i in range(10)]
        tr.update(e1)

        self.assertEqual(len(e1), len(tr))

        # test: initial value of added elements should be 1
        expected = np.ones(len(e1), dtype=np.float64)
        actual = tr.values[tr.indexes(e1)]

        self.assertTrue(np.array_equal(expected, actual))

        # test: adding elements that already exist should not increase the length of trace
        tr.update(e1)

        self.assertEqual(len(e1), len(tr))

        # test: length of trace should only be incremented by the number of new elements
        e2 = np.array([str(i) for i in range(5, 10)])
        tr.update(e2)

        self.assertEqual(len(set(e1).union(e2)), len(tr))

    def test_update_and_decay(self):
        tr = AccumulatingTrace()

        elements = [str(i) for i in range(10)]
        tr.update(elements)

        # test: initial update should set values in active set to one
        expected = np.ones(len(elements), dtype=np.float64)
        actual = tr.values[tr.indexes(elements)]

        self.assertTrue(np.array_equal(expected, actual))

        # test: subsequent updates should increase the values of active set and decay others
        active = [str(i) for i in range(0, 5)]
        inactive = list(set(elements).difference(active))

        tr.update(active)

        self.assertTrue(np.alltrue(tr.values[tr.indexes(active)] > 1.0))
        self.assertTrue(np.alltrue(tr.values[tr.indexes(inactive)] < 1.0))

    def test_update_decay_to_approx_zero(self):
        tr = AccumulatingTrace()

        elements = [str(i) for i in range(10)]
        tr.update(elements)

        for _ in range(30):
            tr.update()

        self.assertTrue(np.allclose(tr.values, 0.0))

    def test_contains(self):
        tr = AccumulatingTrace()

        elements = [str(i) for i in range(10)]
        tr.update(elements)

        for e in elements:
            self.assertIn(e, tr)

        elements_not_in = [str(i) for i in range(10, 20)]
        for e in elements_not_in:
            self.assertNotIn(e, tr)

    def test_expand_allocated(self):
        tr = AccumulatingTrace(pre_allocated=1000, block_size=100)

        # sanity check
        self.assertEqual(1000, tr.n_allocated)

        tr.update([i for i in range(10_000)])

        # test: after update, the number of allocated elements should be 10_000 + one extra block
        self.assertTrue(10_000 + tr.block_size, tr.n_allocated)


class TestReplacingTrace(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

    def test_init(self):
        tr = ReplacingTrace(decay_rate=0.15, pre_allocated=250)

        self.assertEqual(0.15, tr.decay_rate)
        self.assertEqual(250, tr.n_allocated)

    def test_update_with_empty_set(self):
        tr = ReplacingTrace()

        # test: adding empty set should not increment length of trace
        tr.update([])

        self.assertEqual(0, len(tr))

    def test_update_add_single(self):
        tr = ReplacingTrace()

        # test: initial length should be 0
        self.assertEqual(0, len(tr))

        # test: adding a single element should increment length by 1
        tr.update(['1'])
        self.assertEqual(1, len(tr))

        # test: add an element that already exists should not increment the length
        tr.update(['1'])
        self.assertEqual(1, len(tr))

        # test: adding a new element that does not exist should increment length by 1
        tr.update(['2'])
        self.assertEqual(2, len(tr))

    def test_update_add_multiple(self):
        tr = ReplacingTrace()

        # test: adding multiple elements should increment length by the number of elements
        e1 = [str(i) for i in range(10)]
        tr.update(e1)

        self.assertEqual(len(e1), len(tr))

        # test: initial value of added elements should be 1
        expected = np.ones(len(e1), dtype=np.float64)
        actual = tr.values[tr.indexes(e1)]

        self.assertTrue(np.array_equal(expected, actual))

        # test: adding elements that already exist should not increase the length of trace
        tr.update(e1)

        self.assertEqual(len(e1), len(tr))

        # test: length of trace should only be incremented by the number of new elements
        e2 = np.array([str(i) for i in range(5, 10)])
        tr.update(e2)

        self.assertEqual(len(set(e1).union(e2)), len(tr))

    def test_update_and_decay(self):
        decay_rate = 0.25
        tr = ReplacingTrace(decay_rate=decay_rate)

        elements = [str(i) for i in range(10)]
        tr.update(elements)

        # test: initial update should set values in active set to one
        expected = np.ones(len(elements), dtype=np.float64)
        actual = tr.values[tr.indexes(elements)]

        self.assertTrue(np.array_equal(expected, actual))

        # test: subsequent updates should increase the values of active set and decay others
        active = [str(i) for i in range(0, 5)]
        inactive = list(set(elements).difference(active))

        tr.update(active)

        self.assertTrue(np.alltrue(tr.values[tr.indexes(active)] == 1.0))
        self.assertTrue(np.alltrue(tr.values[tr.indexes(inactive)] == decay_rate))

    def test_update_decay_to_approx_zero(self):
        tr = ReplacingTrace()

        elements = [str(i) for i in range(10)]
        tr.update(elements)

        for _ in range(30):
            tr.update()

        self.assertTrue(np.allclose(tr.values, 0.0))

    def test_multiple_decays(self):
        n = 10
        decay_rate = 0.1
        tr = ReplacingTrace(decay_rate=decay_rate)

        elements = [str(i) for i in range(n)]

        # active set decreased by one element per update, which should lead to distinct values from 1e^(-n) to 1.0
        for i in range(len(elements)):
            tr.update(active_set=elements[i:])

        expected_values = list(map(lambda i: decay_rate ** i, range(9, -1, -1)))

        self.assertTrue(np.allclose(tr.values, np.array(expected_values)))

    def test_multiple_updates(self):
        tr = ReplacingTrace()

        elements = [str(i) for i in range(10)]

        # update multiple times
        for _ in range(10):
            tr.update(elements)

        # test: values SHOULD be capped at 1.0, even after multiple updates
        self.assertTrue(np.array_equal(np.ones_like(elements, dtype=np.float64), tr.values))

    def test_contains(self):
        tr = ReplacingTrace()

        elements = [str(i) for i in range(10)]
        tr.update(elements)

        for e in elements:
            self.assertIn(e, tr)

        elements_not_in = [str(i) for i in range(10, 20)]
        for e in elements_not_in:
            self.assertNotIn(e, tr)

    def test_expand_allocated(self):
        tr = ReplacingTrace(pre_allocated=1000, block_size=100)

        # sanity check
        self.assertEqual(1000, tr.n_allocated)

        tr.update([i for i in range(10_000)])

        # test: after update, the number of allocated elements should be 10_000 + one extra block
        self.assertTrue(10_000 + tr.block_size, tr.n_allocated)
