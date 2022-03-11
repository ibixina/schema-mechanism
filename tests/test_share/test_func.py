from collections import Hashable
from copy import copy
from typing import Any

from schema_mechanism.core import GlobalStats
from schema_mechanism.core import ItemPool
from schema_mechanism.share import GlobalParams
from schema_mechanism.share import Verbosity


def common_test_setup():
    ItemPool().clear()

    GlobalParams().reset()
    GlobalParams().set('verbosity', Verbosity.WARN)

    GlobalStats().reset()


def is_eq_reflexive(x: Any) -> bool:
    """ Tests the reflexive property of the __eq__ method.

    For any non-None object x, x == x should return True.

    :param x: any non-None object

    :return: True if x's __eq__ method satisfies the reflexive property for this object instance
    """
    assert (x is not None)

    return x == x


def is_eq_symmetric(x: Any, y: Any) -> bool:
    """ Tests the symmetric property of the __eq__ method.

    For any non-None objects x and y, x == y iff y = x.

    :param x: any non-None object for which x == y
    :param y: any non-None object for which x == y

    :return: True if the __eq__ methods satisfy the symmetric property for these object instances
    """
    assert (x is not None)
    assert (y is not None)
    assert (x == y)

    return y == x


def is_eq_transitive(x: Any, y: Any, z: Any) -> bool:
    """ Tests the transitive property of the __eq__ method.

    For any non-None objects x, y, and z, if x == y and y == z, then x == z.

    :param x: any non-None object equal to y
    :param y: any non-None object equal to y and z
    :param z: any non-None object equal to y

    :return: True if the __eq__ methods satisfy the transitive property for these object instances
    """
    assert (x is not None)
    assert (y is not None)
    assert (z is not None)
    assert (x == y)
    assert (y == z)

    return x == z


def is_eq_consistent(x: Any, y: Any, count: int = 1) -> bool:
    """ Tests the consistency property of the __eq__ method.

    For any non-None, immutable objects x and y, multiple calls to x == y should return the same result.

    :param x: any non-None object
    :param y: any non-None object
    :param count: the number of times to perform the consistency check

    :return: True if the __eq__ methods satisfy the consistency property for these object instances
    """
    assert (x is not None)
    assert (y is not None)

    if x == y:
        return all({x == y for _ in range(count)})
    else:
        return all({x != y for _ in range(count)})


def is_eq_with_null_is_false(x: Any) -> bool:
    """ Tests the null argument property of the __eq__ method.

    For any non-None object x, x == None should return False.

    :param x: any non-None object

    :return: True if x's __eq__ method satisfies the null argument property for this object instance
    """
    assert (x is not None)

    return x != None


def is_hash_consistent(x: Hashable, count: int = 1) -> bool:
    """ Test that the object's hash method consistently returns the same hash.

    :param x: any non-None, hashable object
    :param count: the number of times to perform the consistency check

    :return: True if x's __hash__ method satisfies the consistency property for this object instance
    """
    h_value = hash(x)
    return all({h_value == hash(x) for _ in range(count)})


def is_hash_same_for_equal_objects(x: Hashable, y: Hashable) -> bool:
    """ Test hash method produces same result for equal objects.

    :param x: non-None, hashable object equal to y
    :param y: non-None, hashable object equal to x

    :return: True if x's __hash__ method returns same hash for equal objects
    """
    assert (x == y)

    return hash(x) == hash(y)


def satisfies_equality_checks(obj: Any, other: Any) -> bool:
    copy_ = copy(obj)

    return all({
        is_eq_reflexive(obj),
        is_eq_symmetric(x=obj, y=copy_),
        is_eq_transitive(x=obj, y=copy_, z=copy(copy_)),
        is_eq_consistent(x=obj, y=copy_),
        is_eq_with_null_is_false(obj),
        obj != other
    })


def satisfies_hash_checks(obj: Any) -> bool:
    return all({
        isinstance(hash(obj), int),
        is_hash_consistent(obj),
        is_hash_same_for_equal_objects(obj, obj),
        is_hash_same_for_equal_objects(obj, copy(obj)),
    })
