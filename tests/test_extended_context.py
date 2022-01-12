from random import sample
from unittest import TestCase

from schema_mechanism.data_structures import ExtendedContext
from schema_mechanism.data_structures import ItemPool
from schema_mechanism.data_structures import ItemPoolStateView
from schema_mechanism.data_structures import NULL_EC_ITEM_STATS
from schema_mechanism.data_structures import ReadOnlyItemPool
from schema_mechanism.data_structures import SchemaStats
from schema_mechanism.data_structures import SymbolicItem


class TestExtendedContext(TestCase):
    N_ITEMS = 1000

    def setUp(self) -> None:
        pool = ItemPool()
        pool.clear()

        for i in range(self.N_ITEMS):
            pool.get(i, SymbolicItem)

    def test_init(self):
        ec = ExtendedContext(SchemaStats())
        for i in ItemPool().items:
            self.assertIs(NULL_EC_ITEM_STATS, ec.stats[i])

    def test_update(self):
        state = sample(range(self.N_ITEMS), k=10)

        # in practice, this would be supplied from the schema containing this ExtendedResult
        schema_stats = SchemaStats()

        # simulate schema being activated with action taken
        schema_stats.update(activated=True, success=True)

        ec = ExtendedContext(schema_stats)

        # update an item from this state assuming the action was taken
        ec.update(state[0], is_on=True, success=True)

        item_stats = ec.stats[state[0]]
        for i in ItemPool().items:
            if i.state_element != state[0]:
                self.assertIs(NULL_EC_ITEM_STATS, ec.stats[i])
            else:
                self.assertIsNot(NULL_EC_ITEM_STATS, item_stats)

                for value in [item_stats.n_on,
                              item_stats.n_success_and_on]:
                    self.assertEqual(1, value)

                for value in [item_stats.n_off,
                              item_stats.n_success_and_off,
                              item_stats.n_fail_and_on,
                              item_stats.n_fail_and_off]:
                    self.assertEqual(0, value)

    def test_update_all(self):
        state = sample(range(self.N_ITEMS), k=10)
        view = ItemPoolStateView(pool=ReadOnlyItemPool(), state=state)

        # in practice, this would be supplied from the schema containing this ExtendedResult
        schema_stats = SchemaStats()

        # simulate schema being activated with action taken
        schema_stats.update(activated=True, success=True)

        ec = ExtendedContext(schema_stats)

        # update all items in this state simulating case of action taken
        ec.update_all(view, success=True)

        # test that all items in the state have been updated
        for i in ItemPool().items:
            item_stats = ec.stats[i]
            if i.state_element in state:
                self.assertEqual(1, item_stats.n_on)
                self.assertEqual(1, item_stats.n_success_and_on)

                self.assertEqual(0, item_stats.n_off)
                self.assertEqual(0, item_stats.n_success_and_off)

                self.assertEqual(0, item_stats.n_fail_and_on)
                self.assertEqual(0, item_stats.n_fail_and_off)
            else:
                self.assertEqual(1, item_stats.n_off)
                self.assertEqual(1, item_stats.n_success_and_off)

                self.assertEqual(0, item_stats.n_on)
                self.assertEqual(0, item_stats.n_success_and_on)

                self.assertEqual(0, item_stats.n_fail_and_on)
                self.assertEqual(0, item_stats.n_fail_and_off)
