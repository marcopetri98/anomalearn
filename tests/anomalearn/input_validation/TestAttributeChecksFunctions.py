import unittest

from anomalearn.input_validation import is_var_of_type, check_attributes_exist, \
    check_not_default_attributes, check_argument_types


class CustomObject(object):
    def __init__(self):
        super().__init__()
        
        self.att1 = "Colonel Miles Quaritch"
        self.att2 = "Tonowari"


class TestAttributeChecksFunctions(unittest.TestCase):
    def test_is_var_of_type(self):
        self.assertTrue(is_var_of_type("Neytiri", [int, float, str]))
        self.assertTrue(is_var_of_type(14.02, [int, float, str]))
        self.assertTrue(is_var_of_type(2017, [int, float, str]))
        
        self.assertFalse(is_var_of_type("Neytiri", [int, float]))
    
    def test_check_attributes_exists(self):
        custom_object = CustomObject()
        check_attributes_exist(custom_object, ["att1"])
        check_attributes_exist(custom_object, ["att1", "att2"])
        self.assertRaises(ValueError, check_attributes_exist, custom_object, "Tsireya")
        self.assertRaises(ValueError, check_attributes_exist, custom_object, ["Lo'ak", "Tsireya"])
        self.assertRaises(ValueError, check_attributes_exist, custom_object, ["att1", "Tsireya"])
        self.assertRaises(ValueError, check_attributes_exist, custom_object, ["Lo'ak", "att1"])
    
    def test_check_not_default_attributes(self):
        custom_object = CustomObject()
        check_not_default_attributes(custom_object, {"att1": "Default"})
        check_not_default_attributes(custom_object, {"att1": "Default 1", "att2": "Default 2"})
        self.assertRaises(ValueError, check_not_default_attributes, custom_object, {"att1": "Colonel Miles Quaritch"})
        self.assertRaises(ValueError, check_not_default_attributes, custom_object, {"att2": "Tonowari"})
        self.assertRaises(ValueError, check_not_default_attributes, custom_object, {"att1": "Colonel Miles Quaritch", "att2": "Tonowari"})
        self.assertRaises(ValueError, check_not_default_attributes, custom_object, {"att1": "Default 1", "att2": "Tonowari"})
        self.assertRaises(ValueError, check_not_default_attributes, custom_object, {"att1": "Colonel Miles Quaritch", "att2": "Default 2"})
    
    def test_check_argument_types(self):
        check_argument_types([1], [int])
        check_argument_types([1, 1.22], [int, float])
        check_argument_types([1, 1.22], [int, float], ["first", "second"])
        
        # check value errors
        self.assertRaises(ValueError, check_argument_types, [1], [int, float])
        self.assertRaises(ValueError, check_argument_types, [1, 1.22], [int])
        self.assertRaises(ValueError, check_argument_types, [1, 1.22], [int, float], ["first"])
        self.assertRaises(ValueError, check_argument_types, [1, 1.22], [int], ["first", "second"])
        self.assertRaises(ValueError, check_argument_types, [1], [int, float], ["first", "second"])
        self.assertRaises(ValueError, check_argument_types, [1], [int, float], ["first", "second", "third"])
        
        # check types
        self.assertRaises(TypeError, check_argument_types, [1], [str])
        self.assertRaises(TypeError, check_argument_types, [1], [float])
        self.assertRaises(TypeError, check_argument_types, ["Kiri"], [float])
        self.assertRaises(TypeError, check_argument_types, [1, 1.22], [float, float])
        self.assertRaises(TypeError, check_argument_types, [1, "Kiri"], [int, float])
