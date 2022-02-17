from random import sample
from unittest import TestCase
from unittest.mock import PropertyMock
from unittest.mock import patch

from schema_mechanism.data_structures import ExtendedContext
from schema_mechanism.data_structures import GlobalOption
from schema_mechanism.data_structures import GlobalParams
from schema_mechanism.data_structures import ItemAssertion
from schema_mechanism.data_structures import ItemPool
from schema_mechanism.data_structures import ItemPoolStateView
from schema_mechanism.data_structures import NULL_EC_ITEM_STATS
from schema_mechanism.data_structures import State
from schema_mechanism.data_structures import SymbolicItem
from schema_mechanism.func_api import sym_asserts
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_state
from schema_mechanism.func_api import sym_state_assert
from test_share.test_classes import MockObserver
from test_share.test_func import common_test_setup


class TestExtendedContext(TestCase):
    N_ITEMS = 1000

    def setUp(self) -> None:
        common_test_setup()

        pool = ItemPool()

        for i in range(self.N_ITEMS):
            pool.get(str(i), item_type=SymbolicItem)

        self.context = sym_state_assert('100,101')
        self.ec = ExtendedContext(context=self.context)

        self.obs = MockObserver()
        self.ec.register(self.obs)

    def test_init(self):
        ec = ExtendedContext(self.context)
        for i in ItemPool():
            self.assertIs(NULL_EC_ITEM_STATS, ec.stats[i])

        for ia in self.context:
            self.assertIn(ia.item, ec.suppress_list)

    def test_update(self):
        state = sample(range(self.N_ITEMS), k=10)

        # update an item from this state assuming the action was taken
        new_item = sym_item(state[0])
        self.ec.update(new_item, on=True, success=True)

        item_stats = self.ec.stats[new_item]
        for i in ItemPool():
            if i.state_element != state[0]:
                self.assertIs(NULL_EC_ITEM_STATS, self.ec.stats[i])
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

    def test_update_all_1(self):
        state = State(sample(range(self.N_ITEMS), k=10))
        view = ItemPoolStateView(state)

        # update all items in this state simulating case of action taken
        self.ec.update_all(view, success=True)

        # test that all items in the state have been updated
        for i in ItemPool():
            item_stats = self.ec.stats[i]

            if i.state_element in state:
                self.assertEqual(1, item_stats.n_on)
                self.assertEqual(1, item_stats.n_success_and_on)

                self.assertEqual(0, item_stats.n_off)
                self.assertEqual(0, item_stats.n_success_and_off)

                self.assertEqual(0, item_stats.n_fail_and_on)
                self.assertEqual(0, item_stats.n_fail_and_off)
            else:

                self.assertEqual(0, item_stats.n_on)
                self.assertEqual(0, item_stats.n_success_and_on)

                self.assertEqual(1, item_stats.n_off)
                self.assertEqual(1, item_stats.n_success_and_off)

                self.assertEqual(0, item_stats.n_fail_and_on)
                self.assertEqual(0, item_stats.n_fail_and_off)

    def test_update_all_2(self):
        # test update_all without GlobalOption.EC_MOST_SPECIFIC_ON_MULTIPLE enabled
        if GlobalOption.EC_MOST_SPECIFIC_ON_MULTIPLE in GlobalParams().options:
            GlobalParams().options.remove(GlobalOption.EC_MOST_SPECIFIC_ON_MULTIPLE)

        self.ec.update_all(ItemPoolStateView(sym_state('3')), success=False)
        self.ec.update_all(ItemPoolStateView(sym_state('2')), success=False, count=5)

        self.assertEqual(0, len(self.ec.relevant_items))

        # three pending relevant items from this update:
        # [1 -> SC=1.0; FC=0.0]   [2 -> SC=1.0; FC=0.25]   [3 -> SC=0.0; FC=0.75]
        self.ec.update_all(ItemPoolStateView(sym_state('1,2')), success=True, count=10)

        # test: all 3 pending items should be relevant items in extended context
        self.assertEqual(3, len(self.ec.relevant_items))
        self.assertSetEqual(set(sym_asserts('1,2,~3')), self.ec.relevant_items)

    def test_update_all_3(self):
        # test update_all with GlobalOption.EC_MOST_SPECIFIC_ON_MULTIPLE enabled
        GlobalParams().options.add(GlobalOption.EC_MOST_SPECIFIC_ON_MULTIPLE)

        self.ec.update_all(ItemPoolStateView(sym_state('3')), success=False)
        self.ec.update_all(ItemPoolStateView(sym_state('2')), success=False, count=5)

        self.assertEqual(0, len(self.ec.relevant_items))

        # three pending relevant items from this update:
        # [1 -> SC=1.0; FC=0.0]   [2 -> SC=1.0; FC=0.25]   [3 -> SC=0.0; FC=0.75]
        self.ec.update_all(ItemPoolStateView(sym_state('1,2')), success=True, count=10)

        # test: only a single relevant item should be added to the extended context
        self.assertEqual(1, len(self.ec.relevant_items))
        self.assertSetEqual(set(sym_asserts('~3')), self.ec.relevant_items)

    def test_defer_update_to_spin_offs(self):
        # GlobalOption.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA enabled
        GlobalParams().options.add(GlobalOption.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA)

        update_view_1 = ItemPoolStateView(sym_state('1,3,4'))
        update_view_2 = ItemPoolStateView(sym_state('1,6'))
        defer_view_1 = ItemPoolStateView(sym_state('1,5'))
        defer_view_2 = ItemPoolStateView(sym_state('1,7,8'))

        with patch(target='schema_mechanism.data_structures.ExtendedContext.relevant_items',
                   new_callable=PropertyMock) as mock_relevant_items:
            mock_relevant_items.return_value = set(sym_asserts('5,~6,7'))
            ec = ExtendedContext(sym_state_assert('1,~2'))

            self.assertFalse(ec.defer_update_to_spin_offs(update_view_1))
            self.assertFalse(ec.defer_update_to_spin_offs(update_view_2))
            self.assertTrue(ec.defer_update_to_spin_offs(defer_view_1))
            self.assertTrue(ec.defer_update_to_spin_offs(defer_view_2))

    def test_register_and_unregister(self):
        observer = MockObserver()
        self.ec.register(observer)
        self.assertIn(observer, self.ec.observers)

        self.ec.unregister(observer)
        self.assertNotIn(observer, self.ec.observers)

    def test_relevant_items_1(self):
        items = [sym_item(str(i)) for i in range(5)]

        i1 = items[0]

        # test updates that SHOULD NOT result in relevant item determination
        self.ec.update(i1, on=False, success=True)

        self.assertEqual(0, len(self.ec.pending_relevant_items))
        self.ec.update(i1, on=False, success=False, count=2)
        self.assertEqual(0, len(self.ec.pending_relevant_items))

        self.ec.update(i1, on=True, success=False)
        self.assertEqual(0, len(self.ec.pending_relevant_items))

        self.ec.update(i1, on=True, success=True)
        self.assertEqual(0, len(self.ec.pending_relevant_items))

        # test update that SHOULD result in a relevant item
        self.ec.update(i1, on=True, success=True)

        i1_stats = self.ec.stats[i1]
        self.assertTrue(i1_stats.success_corr > self.ec.POS_CORR_RELEVANCE_THRESHOLD)
        self.assertTrue(i1_stats.failure_corr <= self.ec.NEG_CORR_RELEVANCE_THRESHOLD)

        self.assertEqual(1, len(self.ec.pending_relevant_items))
        self.ec.check_pending_relevant_items()
        self.assertEqual(0, len(self.ec.pending_relevant_items))

        # verify only one relevant item
        self.assertEqual(1, len(self.ec.relevant_items))
        self.assertIn(ItemAssertion(i1, negated=False), self.ec.relevant_items)

        # should add a 2nd relevant item
        i2 = items[1]

        self.ec.update(i2, on=True, success=True)
        self.ec.update(i2, on=False, success=False)

        self.assertEqual(1, len(self.ec.pending_relevant_items))
        self.ec.check_pending_relevant_items()
        self.assertEqual(0, len(self.ec.pending_relevant_items))

        self.assertEqual(2, len(self.ec.relevant_items))
        self.assertIn(ItemAssertion(i2), self.ec.relevant_items)

        # number of new relevant items SHOULD be reset to zero after notifying observers
        self.ec.notify_all()
        self.assertEqual(0, len(self.ec.new_relevant_items))

    def test_relevant_items_2(self):
        i1 = sym_item('100')

        self.assertIn(i1, self.ec.suppress_list)

        # test updates that SHOULD NOT result in relevant item determination
        self.ec.update(i1, on=True, success=True, count=10)
        self.ec.update(i1, on=False, success=False, count=10)

        i1_stats = self.ec.stats[i1]
        self.assertTrue(i1_stats.success_corr > self.ec.POS_CORR_RELEVANCE_THRESHOLD)
        self.assertTrue(i1_stats.failure_corr <= self.ec.NEG_CORR_RELEVANCE_THRESHOLD)

        # verify suppressed item NOT in relevant items list
        self.assertEqual(0, len(self.ec.pending_relevant_items))
        self.assertEqual(0, len(self.ec.relevant_items))
        self.assertNotIn(ItemAssertion(i1), self.ec.relevant_items)
