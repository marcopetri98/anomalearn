from enum import Enum


class MissingStrategy(Enum):
    NOTHING = 0,
    DROP = 1,
    INTERPOLATION = 2,
    FIXED_VALUE = 3
