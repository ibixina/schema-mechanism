from abc import ABC
from collections import deque
from typing import Any
from typing import Optional

from schema_mechanism.core import Action
from schema_mechanism.core import CompositeItem
from schema_mechanism.core import GlobalStats
from schema_mechanism.core import Item
from schema_mechanism.core import Schema
from schema_mechanism.core import StateAssertion
from schema_mechanism.core import SymbolicItem
from schema_mechanism.modules import PendingDetails
from schema_mechanism.modules import SchemaSelection
from schema_mechanism.util import Observable
from schema_mechanism.util import Observer


class MockObserver(Observer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.messages = []

    @property
    def n_received(self) -> int:
        return len(self.messages)

    @property
    def last_message(self) -> dict[str, Any]:
        return {} if not self.messages else self.messages[-1]

    def receive(self, *args, **kwargs) -> None:
        self.messages.append({'args': args, 'kwargs': kwargs})


class MockObservable(Observable):
    pass


class MockItem(Item, ABC):
    def __init__(self,
                 source: Any,
                 primitive_value: Optional[float] = None,
                 avg_accessible_value: Optional[float] = None,
                 delegated_value: Optional[float] = None,
                 **kwargs):
        super().__init__(source, primitive_value, **kwargs)

        self._mock_avg_accessible_value = avg_accessible_value
        self._mock_delegated_value = delegated_value


class MockSymbolicItem(SymbolicItem):
    def __init__(self,
                 source: str,
                 primitive_value: Optional[float] = None,
                 avg_accessible_value: Optional[float] = None,
                 delegated_value: Optional[float] = None,
                 **kwargs):
        super().__init__(source, primitive_value, **kwargs)

        self._mock_avg_accessible_value = avg_accessible_value
        self._mock_delegated_value = delegated_value

    @property
    def avg_accessible_value(self) -> float:
        return (
            super().avg_accessible_value
            if self._mock_avg_accessible_value is None
            else self._mock_avg_accessible_value
        )

    @avg_accessible_value.setter
    def avg_accessible_value(self, value: float) -> None:
        self._mock_avg_accessible_value = value

    @property
    def delegated_value(self) -> float:
        if self._mock_delegated_value is not None:
            return self._mock_delegated_value

        return self.avg_accessible_value - GlobalStats().baseline_value

    @delegated_value.setter
    def delegated_value(self, value) -> None:
        self._mock_delegated_value = value


class MockCompositeItem(CompositeItem):
    def __init__(self,
                 source: StateAssertion,
                 avg_accessible_value: float,
                 delegated_value: Optional[float] = None,
                 **kwargs):
        super().__init__(source, **kwargs)

        self._mock_avg_accessible_value = avg_accessible_value
        self._mock_delegated_value = delegated_value

    @property
    def avg_accessible_value(self) -> float:
        return (
            super().avg_accessible_value
            if self._mock_avg_accessible_value is None
            else self._mock_avg_accessible_value
        )

    @avg_accessible_value.setter
    def avg_accessible_value(self, value: float) -> None:
        self._mock_avg_accessible_value = value

    @property
    def delegated_value(self) -> float:
        if self._mock_delegated_value is not None:
            return self._mock_delegated_value

        return self.avg_accessible_value - GlobalStats().baseline_value

    @delegated_value.setter
    def delegated_value(self, value) -> None:
        self._mock_delegated_value = value


class MockSchema(Schema):
    def __init__(self,
                 action: Action,
                 context: Optional[StateAssertion] = None,
                 result: Optional[StateAssertion] = None,
                 reliability: float = None,
                 avg_duration: float = None,
                 cost: float = None,
                 **kwargs):
        super().__init__(action=action, context=context, result=result, **kwargs)

        self._mock_reliability = reliability
        self._mock_avg_duration = avg_duration
        self._mock_cost = cost

    @property
    def reliability(self) -> float:
        return (
            super().reliability
            if self._mock_reliability is None
            else self._mock_reliability
        )

    @property
    def avg_duration(self) -> float:
        return (
            super().avg_duration
            if self._mock_avg_duration is None
            else self._mock_avg_duration
        )

    @property
    def cost(self) -> float:
        return (
            super().cost
            if self._mock_cost is None
            else self._mock_cost
        )


class MockSchemaSelection(SchemaSelection):
    def __init__(self, pending_schemas: Optional[deque[PendingDetails]] = None, **kwargs):
        super().__init__(**kwargs)

        self._pending_details = pending_schemas
