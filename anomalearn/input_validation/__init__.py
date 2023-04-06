from .array_checks import check_array_1d
from .array_checks import check_array_general
from .array_checks import check_x_y_smaller_1d

from .attribute_checks import check_argument_types
from .attribute_checks import check_attributes_exist
from .attribute_checks import check_not_default_attributes
from .attribute_checks import is_var_of_type

from .colors import is_matplotlib_color

__all__ = ["check_array_1d", "check_array_general", "check_x_y_smaller_1d",
           "check_argument_types", "check_attributes_exist", "is_var_of_type",
           "check_not_default_attributes", "is_matplotlib_color"]
