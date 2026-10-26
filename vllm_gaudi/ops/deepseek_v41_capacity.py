# SPDX-License-Identifier: Apache-2.0
"""Shared capacity contract for DSpark request and packed-row owners.

Serial B1 stays available for backwards compatibility. DSpark concurrency is
an arbitrary integer in ``[MIN_DSV41_DSPARK_CAPACITY,
MAX_DSV41_DSPARK_CAPACITY]``; the upper bound matches the existing V4.1
``max_num_seqs`` limit rather than selecting a particular deployment shape.
"""

MIN_DSV41_DSPARK_CAPACITY = 2
MAX_DSV41_DSPARK_CAPACITY = 64

# Native row-oriented DSV4.1 operators expose a 64-row ABI. Python callers
# tile target/draft packs across this bound while keeping request capacity N.
MAX_DSV41_NATIVE_STATE_ROWS = 64


def validate_dspark_capacity(value, *, allow_serial=False, name="DSpark capacity"):
    """Return a validated integer capacity or raise without coercion.

    Internal request-state helpers keep accepting B1 when ``allow_serial`` is
    true. Concurrent DSpark admission and its launchers use the default
    2..64 contract.
    """
    minimum = 1 if allow_serial else MIN_DSV41_DSPARK_CAPACITY
    if type(value) is not int or not minimum <= value <= MAX_DSV41_DSPARK_CAPACITY:
        raise ValueError(
            f"{name} must be an integer in {minimum}..{MAX_DSV41_DSPARK_CAPACITY}"
        )
    return value


__all__ = [
    "MAX_DSV41_DSPARK_CAPACITY",
    "MAX_DSV41_NATIVE_STATE_ROWS",
    "MIN_DSV41_DSPARK_CAPACITY",
    "validate_dspark_capacity",
]
