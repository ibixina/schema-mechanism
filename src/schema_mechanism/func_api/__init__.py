from __future__ import annotations

from collections.abc import Collection
from typing import Optional

import lark.exceptions

from schema_mechanism.core import Action
from schema_mechanism.core import Assertion
from schema_mechanism.core import Item
from schema_mechanism.core import ItemAssertion
from schema_mechanism.core import Schema
from schema_mechanism.core import SchemaTreeNode
from schema_mechanism.core import State
from schema_mechanism.core import StateAssertion
from schema_mechanism.core import lost_state
from schema_mechanism.core import new_state
from schema_mechanism.func_api.parser import parse


# TODO: generalize the current string -> int conversions in the sym_* methods to support other conversion types
def sym_state(str_repr: str, label: Optional[str] = None) -> State:
    if not str_repr:
        return State([], label)
    return State([se for se in str_repr.split(',')], label)


def sym_item(str_repr: str, **kwargs) -> Item:
    try:
        obj = parse(str_repr, **kwargs)
    except lark.exceptions.UnexpectedInput:
        raise ValueError(f'String representation for item is invalid: {str_repr}')
    assert isinstance(obj, Item)
    return obj


def sym_items(str_repr: str, primitive_values: Collection[float] = None, **kwargs) -> Collection[Item]:
    state_elements = str_repr.split(',')
    if primitive_values:
        if len(state_elements) != len(primitive_values):
            raise ValueError('Primitive values must either be None or one must be given for each state element')
        return [sym_item(se, primitive_value=val, **kwargs) for se, val in zip(state_elements, primitive_values)]
    return [sym_item(token, **kwargs) for token in str_repr.split(',')]


def sym_item_assert(str_repr: str, **kwargs) -> ItemAssertion:
    obj = sym_assert(str_repr, **kwargs)
    if not isinstance(obj, ItemAssertion):
        raise ValueError(f'String representation for item assertion is invalid: {str_repr}')
    return obj


def sym_state_assert(str_repr: str, **kwargs) -> StateAssertion:
    obj = sym_assert(str_repr, **kwargs)
    if not isinstance(obj, StateAssertion):
        raise ValueError(f'String representation for state assertion is invalid: {str_repr}')
    return obj


def sym_assert(str_repr: str, **kwargs) -> Assertion:
    try:
        obj = parse(str_repr, **kwargs)
    except lark.exceptions.UnexpectedInput:
        raise ValueError(f'String representation for assertion is invalid: {str_repr}')

    # workaround to a parsing ambiguity when single character strings are passed (parser currently returns an Item)
    if isinstance(obj, Item):
        obj = ItemAssertion(item=obj)

    assert isinstance(obj, Assertion)
    return obj


def sym_asserts(str_repr: str, **kwargs) -> Collection[Assertion]:
    return [sym_assert(token, **kwargs) for token in str_repr.split(',')]


def sym_schema_tree_node(str_repr: str, label: str = None) -> SchemaTreeNode:
    context_str, action_str, _ = str_repr.split('/')
    return SchemaTreeNode(
        context=sym_assert(context_str) if context_str else None,
        action=Action(action_str) if action_str else None,
        label=label
    )


def sym_schema(str_repr: str, **kwargs) -> Schema:
    try:
        obj = parse(str_repr, **kwargs)
    except lark.exceptions.UnexpectedInput:
        raise ValueError(f'String representation for schema is invalid: {str_repr}')

    assert isinstance(obj, Schema)
    return obj


def actions(n: Optional[int] = None, labels: Optional[list] = None) -> Collection[Action]:
    return (
        [Action(label) for label in labels] if labels else
        [Action(str(i)) for i in range(1, n + 1)] if n else
        []
    )


def primitive_schemas(actions_: Collection[Action]) -> tuple[Schema]:
    return tuple([Schema(action=a) for a in actions_])


def update_schema(schema: Schema,
                  activated: bool,
                  s_prev: Optional[State],
                  s_curr: State,
                  explained: Optional[bool] = None,
                  count: int = 1) -> Schema:
    """ Update the schema based on the previous and current state.

    :param schema: the schema to update
    :param activated: a bool indicated whether the schema was activated (explicitly or implicitly)
    :param s_prev: a collection containing the previous state elements
    :param s_curr: a collection containing the current state elements
    :param explained: True if a reliable schema was activated that "explained" the last state transition
    :param count: the number of times to perform this update

    :return: the updated schema
    """
    schema.update(activated=activated,
                  s_prev=s_prev,
                  s_curr=s_curr,
                  new=new_state(s_prev, s_curr),
                  lost=lost_state(s_prev, s_curr),
                  explained=explained,
                  count=count)

    return schema
