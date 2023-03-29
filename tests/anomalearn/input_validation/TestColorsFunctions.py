import unittest

import matplotlib

from anomalearn.input_validation import is_matplotlib_color


class TestColorsFunctions(unittest.TestCase):
    def test_is_matplotlib_color(self):
        colors = matplotlib.colors
        color_names = set().union(colors.BASE_COLORS.keys())
        color_names = color_names.union(colors.CSS4_COLORS.keys())
        color_names = color_names.union(colors.XKCD_COLORS.keys())
        color_names = color_names.union(colors.TABLEAU_COLORS.keys())
        color_values = set().union(colors.BASE_COLORS.values())
        color_values = color_values.union(colors.CSS4_COLORS.values())
        color_values = color_values.union(colors.XKCD_COLORS.values())
        color_values = color_values.union(colors.TABLEAU_COLORS.values())
        
        for color in color_names:
            self.assertTrue(is_matplotlib_color(color))
        for color in color_values:
            self.assertTrue(is_matplotlib_color(color))
        
        color_names_list = [list(color_names)[e] for e in range(10)]
        color_values_list = [list(color_values)[e] for e in range(10)]
        
        self.assertTrue(is_matplotlib_color(color_names_list))
        self.assertTrue(is_matplotlib_color(color_values_list))
        
        self.assertFalse(is_matplotlib_color("Jake Sully"))
        self.assertFalse(is_matplotlib_color(object()))
        self.assertFalse(is_matplotlib_color(1235.21))
