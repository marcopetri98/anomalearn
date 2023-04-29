import argparse
import ast
import importlib.util
import inspect
import os
import re
import sys
from abc import ABC
from pathlib import Path
from typing import Callable

MODULE_KEY = "modules"
PACKAGES_KEY = "pure_packages"


def process_arguments(argv) -> dict:
    """Process arguments passed to the script.

    Parameters
    ----------
    argv
        Arguments passed to the script when it is run as main.

    Returns
    -------
    options : dict
        It is a dictionary containing all the options of the script, if they are
        not passed to the program they have None as value.
    """
    parser = argparse.ArgumentParser(prog="doc-creator.py",
                                     description="Create and generate docs")

    parser.add_argument("-s", "--source",
                        action="store",
                        default="./docs/source",
                        help="the folder in which the source of the docs is "
                             "located. It defaults to \"./docs/source\".")
    parser.add_argument("-a", "--api-location",
                        action="store",
                        default="./reference/api",
                        help="relative location of api reference in the docs "
                             "source. It defaults to \"./reference/api\".")
    parser.add_argument("--max-depth",
                        action="store",
                        type=int,
                        default=2,
                        help="the maximum depth of toc trees. The default is "
                             "2.")
    parser.add_argument("-c", "--complete-package-title",
                        action="store_const",
                        const=True,
                        default=False,
                        help="if given, the documentation of packages and "
                             "modules will have as title the complete python "
                             "package name (e.g. pkg.subpkg.module module). "
                             "Otherwise, only the name of the package will be "
                             "in the title.")
    parser.add_argument("-p", "--private-packages-modules",
                        action="store_const",
                        const=True,
                        default=False,
                        help="if given, also private packages and modules will "
                             "be retrieved from the package and included in the"
                             " documentation. Otherwise, they won't be "
                             "retrieved.")
    parser.add_argument("-P", "--private-members",
                        action="store_const",
                        const=True,
                        default=False,
                        help="if given, both the private and public members of "
                             "retrieved packages will be included in the "
                             "documentation. Otherwise, they won't be included.")
    parser.add_argument("-d", "--dunders",
                        action="store_const",
                        const=True,
                        default=False,
                        help="if given, dunders method that are overridden will"
                             " be included in the documentation. Otherwise, "
                             "they won't be included.")
    parser.add_argument("-i", "--show-inheritance",
                        action="store_const",
                        const=True,
                        default=False,
                        help="if given, the base classes from which the class "
                             "inherits will be shown.")
    parser.add_argument("--show-inherited-members",
                        action="store_const",
                        const=True,
                        default=False,
                        help="if given, the inherited attributes and methods of"
                             " a class will be included in the documentation.")
    parser.add_argument("--inherited-members-before",
                        action="store_const",
                        const=True,
                        default=False,
                        help="if given, the inherited attributes and methods of"
                             " a class will be included in the documentation "
                             "and will be placed before the not inherited "
                             "attributes and methods.")
    parser.add_argument("-v", "--show-attribute-value",
                        action="store_const",
                        const=True,
                        default=False,
                        help="if given, the attributes will contain the default"
                             " value in the documentation.")
    parser.add_argument("--mod-separate",
                        action="store_const",
                        const=True,
                        default=False,
                        help="if given, the contents of a module will be in a "
                             "different page with respect to that of the module"
                             " and will be linked. Otherwise, the content of a "
                             "module is inside the module doc.")
    parser.add_argument("package_location",
                        metavar="PACKAGE",
                        help="it is the relative location in which the package "
                             "containing the code to be documented is located.")

    output = parser.parse_args(argv[1:])
    arguments = vars(output)

    given_source = Path(arguments["source"])
    given_api = Path(arguments["api_location"])
    
    if not given_source.is_dir():
        raise ValueError("The given source is not a directory.")
    if (not given_api.is_dir() and given_api.exists()) or given_api.is_absolute():
        raise ValueError("The given api location is not a directory or is an "
                         "absolute path.")

    return arguments


def get_package_structure(package_path: Path, private: bool) -> dict:
    """Gets the structure of the package.
    
    Parameters
    ----------
    package_path : Path
        The path of the package.
        
    private : bool
        Whether the private packages and modules must be explored.

    Returns
    -------
    structure : dict
        The structure of the package in which each package has a dictionary with
        three keys: modules, packages and namespace packages.
    """
    structure = dict()
    
    package_name = package_path.name
    elements = list(package_path.iterdir())
    
    # get modules
    modules = list(map(lambda x: x[:-3],
                       filter(lambda x: ".py" in x,
                              map(str,
                                  [e.name for e in elements]))))
    if not private:
        modules = list(filter(lambda x: not x.startswith("_") or x == "__init__",
                              modules))
    structure[package_name] = dict()
    structure[package_name][MODULE_KEY] = modules
    
    # get packages
    packages = list(filter(lambda x: x.is_dir() and x.name != "__pycache__",
                           elements))
    if not private:
        packages = list(filter(lambda x: not str(x.name).startswith("_"),
                               packages))
        
    # get pure packages and namespace packages
    pure_packages = []
    for pkg in packages:
        contained = [e.name for e in pkg.iterdir()]
        if "__init__.py" in contained:
            pure_packages.append(pkg)
    
    structure[package_name][PACKAGES_KEY] = []
    for pkg in pure_packages:
        structure[package_name][PACKAGES_KEY].append(get_package_structure(pkg, private))
    
    return structure


def populate_attribute(spaces: str, attr_options: dict) -> str:
    """Populate the attributes depending on the options passed at runtime.
    
    Parameters
    ----------
    spaces : str
        The spaces before the autodoc "autodata" or "autoattribute" directive.
    
    attr_options : dict
        A dictionary containing the options to customize the look and feel of
        attributes in docs.

    Returns
    -------
    rst_content_to_add : str
        The content to add to the rst for populating the attribute doc.
    """
    attribute_content = ""

    if not attr_options["show_value"]:
        attribute_content += spaces + "   :no-value:\n"
    
    return attribute_content


def populate_function(spaces: str, func_options: dict) -> str:
    """Populate the attributes depending on the options passed at runtime.
    
    Parameters
    ----------
    spaces : str
        The spaces before the autodoc "autodata" or "autoattribute" directive.
    
    func_options : dict
        A dictionary containing the options to customize the look and feel of
        functions and methods in docs.

    Returns
    -------
    rst_content_to_add : str
        The content to add to the rst for populating the function doc.
    """
    function_content = ""
    
    return function_content


def is_stdlib_module(path: str | None) -> bool:
    """Checks if the path to the module is that of a stdlib object.
    
    Parameters
    ----------
    path : str | None
        The path found by inspect object of the module. If str, it must be
        converted to standard posix.

    Returns
    -------
    is_stdlib : bool
        True if the path represent that of a stdlib object.
    """
    python_path = Path(inspect.getsourcefile(ABC)).parent.as_posix()
    
    if path is not None:
        path = Path(path).as_posix()

    if path is None or path == "built-in" or path.startswith(python_path):
        return True
    
    return False


def get_leftmost_attribute(attr: ast.Attribute) -> ast.Attribute:
    """Gets the leftmost attribute if any.
    
    Parameters
    ----------
    attr : ast.Attribute
        The attribute to inspect.

    Returns
    -------
    attribute : ast.Attribute
        The leftmost attribute if the attribute contains another attribute as
        value. Otherwise, the passed attribute.
    """
    child = attr.value
    while isinstance(child, ast.Attribute):
        attr = child
        child = child.value
    return attr


def get_class_members(
    class_path: Path,
    class_name: str,
    module_complete_name: str
) -> dict:
    """Gets all the members of the class.
    
    Parameters
    ----------
    class_path : Path
        The path of the class file or string to populate.
        
    class_name : str
        The name of the class to populate.
        
    module_complete_name : str
        The complete name of the module from its root.

    Returns
    -------
    class_members : dict
        The members of the class, inherited from not standard library classes
        and the ones defined or overridden by the class. The keys of the
        dictionary are: attributes, docs_attributes, methods, properties,
        class_methods, decorated_methods and inherited. Each key's value is a
        list of names, except of inherited, which is a dictionary with keys:
        attributes, methods, properties, class_methods and decorated_methods.
        The value of these keys is a list of names. In such a way a complete
        description of the class is provided, and it is possible to distinguish
        members of the analysed class and those which are inherited. The
        difference between attributes and docs_attributes is that the former are
        attributes defined in __init__ and are not described inside the class'
        docstring. The latter are attributes which are described inside the
        class' docstring.
        
    Raises
    ------
    ValueError
        If the class is a standard library class.
    """
    members = dict()
    
    spec = importlib.util.spec_from_file_location(module_complete_name,
                                                  class_path)
            
    if is_stdlib_module(Path(spec.origin).as_posix()):
        raise ValueError("get_class_members must be called on classes that "
                         "are not part of the standard library")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    class_object = module.__dict__[class_name]
    mro = inspect.getmro(class_object)
    bases = mro[1:]

    # without deletion of the key there will be circular dependencies, mainly
    # due to the fact that init imports a module and the module imports
    # something else from init. This behaviour is solved only with standard
    # import statement, or with importlib.import_module. Which cannot be used
    # since the library may not be installed yet and the current working
    # directory might be different from the folder containing the library.
    sys.modules[module_complete_name] = module
    class_source = inspect.getsource(class_object)
    del sys.modules[module_complete_name]
    tree = ast.parse(class_source)
    class_tree = tree.body[0]
    
    attributes = []
    docs_attributes = []
    methods = []
    properties = []
    class_methods = []
    decorated_methods = []
    
    # first iteration with a basic division of roles
    for node in ast.iter_child_nodes(class_tree):
        if isinstance(node, ast.FunctionDef):
            methods.append(node)
        
        # here we search for class attributes or any type of method. All class
        # methods are in the form of foo = something, bar : annotation, or
        # baz : annotation = something
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            if isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(node.target, ast.Name):
                attributes.append(node.target.id)
            else:
                for target in node.targets:
                    if isinstance(target, ast.Tuple):
                        for el in target.elts:
                            if isinstance(el, ast.Name):
                                attributes.append(el.id)
                    
                    if isinstance(target, ast.Name):
                        attributes.append(target.id)
    
    inits = list(filter(lambda x: x.name == "__init__", methods))
    if len(inits):
        init_met = inits[0]

        for child in ast.walk(init_met):
            attr = None
            
            # all attributes are in the form of self.foo = something or in the
            # form of self.bar : annotation = something or in the form of
            # self.baz : annotation
            if isinstance(child, ast.Assign):
                for lhs in child.targets:
                    if isinstance(lhs, ast.Attribute) and isinstance(lhs.value, ast.Name):
                        if lhs.value.id == "self":
                            attr = lhs.attr

            if isinstance(child, ast.AnnAssign):
                if isinstance(child.target, ast.Attribute) and isinstance(child.target.value, ast.Name):
                    if child.target.value.id == "self":
                        attr = child.target.attr
                        
            if attr is not None:
                docstring = inspect.getdoc(class_object)
                
                if attr in docstring:
                    docs_attributes.append(attr)
                else:
                    attributes.append(attr)

    attributes = list(sorted(set(attributes)))
    docs_attributes = list(sorted(set(docs_attributes)))
    
    # divide methods, class methods, properties and generally decorated methods
    for met in methods:
        property_or_class = False
        
        if len(met.decorator_list) > 0:
            for decorator in met.decorator_list:
                if isinstance(decorator, ast.Call):
                    decorator = decorator.func
                
                if isinstance(decorator, ast.Attribute):
                    lhs_attr = get_leftmost_attribute(decorator)
                    if lhs_attr.value.id == met.name and lhs_attr.attr == "setter":
                        name = f"{met.name}.setter"
                    else:
                        name = lhs_attr.value.id
                else:
                    name = decorator.id

                if name == "property" or name == f"{met.name}.setter":
                    properties.append(met.name)
                    property_or_class = True
                    break
                if name == "classmethod":
                    class_methods.append(met.name)
                    property_or_class = True
                    break
        
            if not property_or_class:
                decorated_methods.append(met.name)
    
    properties = list(sorted(set(properties)))
    class_methods = list(sorted(set(class_methods)))
    decorated_methods = list(sorted(set(decorated_methods)))
    
    # clean methods from non-pure methods
    methods = [e.name for e in methods]
    methods = list(sorted(set(methods).difference(properties + class_methods + decorated_methods)))
    
    members["attributes"] = attributes
    members["docs_attributes"] = docs_attributes
    members["methods"] = methods
    members["properties"] = properties
    members["class_methods"] = class_methods
    members["decorated_methods"] = decorated_methods
    
    # get the inherited values
    inherited_attributes = []
    inherited_methods = []
    inherited_properties = []
    inherited_class_methods = []
    inherited_decorated_methods = []
    inherited_dicts = []
    for base in bases:
        if base.__module__ == class_object.__module__:
            inherited_dicts.append(get_class_members(class_path,
                                                     base.__name__,
                                                     module_complete_name))
        else:
            top_lvl_base = base.__module__.split(".", maxsplit=1)[0]
            top_lvl_class = module_complete_name.split(".", maxsplit=1)[0]
            
            # if the derived package is part of the documented library, inspect
            # getmodule won't work. Therefore, the location and spec for the
            # module must be retrieved directly
            if top_lvl_base == top_lvl_class:
                parents = module_complete_name.count(".")
                base_folder = class_path.parents[parents - 1]
                splits = base.__module__.split(".")
                navigation = splits[1:-1]
                for folder in navigation:
                    base_folder /= folder
                base_folder /= splits[-1] + ".py"

                spec = importlib.util.spec_from_file_location(base.__module__,
                                                              base_folder)
                module = importlib.util.module_from_spec(spec)
            else:
                module = inspect.getmodule(base)
            
            if module is not None and not is_stdlib_module(Path(module.__spec__.origin).as_posix()):
                inherited_dicts.append(get_class_members(Path(module.__spec__.origin),
                                                         base.__name__,
                                                         base.__module__))
    
    # collapse all inherited values to a single list
    if len(inherited_dicts) > 0:
        for base in inherited_dicts:
            inherited_attributes.extend(base["attributes"])
            inherited_attributes.extend(base["docs_attributes"])
            inherited_attributes.extend(base["inherited"]["attributes"])
            inherited_methods.extend(base["methods"])
            inherited_methods.extend(base["inherited"]["methods"])
            inherited_properties.extend(base["properties"])
            inherited_properties.extend(base["inherited"]["properties"])
            inherited_class_methods.extend(base["class_methods"])
            inherited_class_methods.extend(base["inherited"]["class_methods"])
            inherited_decorated_methods.extend(base["decorated_methods"])
            inherited_decorated_methods.extend(base["inherited"]["decorated_methods"])
    
    inherited_attributes = list(filter(lambda x: not x.startswith("__"), inherited_attributes))
    inherited_methods = list(filter(lambda x: not x.startswith("__"), inherited_methods))
    inherited_properties = list(filter(lambda x: not x.startswith("__"), inherited_properties))
    inherited_class_methods = list(filter(lambda x: not x.startswith("__"), inherited_class_methods))
    inherited_decorated_methods = list(filter(lambda x: not x.startswith("__"), inherited_decorated_methods))
    
    this_class_methods = set(methods).union(properties).union(decorated_methods)
    
    inherited_attributes = list(sorted(set(inherited_attributes).difference(attributes + docs_attributes)))
    inherited_methods = list(sorted(set(inherited_methods).difference(methods)))
    inherited_properties = list(sorted(set(inherited_properties).difference(this_class_methods)))
    inherited_class_methods = list(sorted(set(inherited_class_methods).difference(class_methods)))
    inherited_decorated_methods = list(sorted(set(inherited_decorated_methods).difference(this_class_methods)))
    
    members["inherited"] = dict()
    members["inherited"]["attributes"] = inherited_attributes
    members["inherited"]["methods"] = inherited_methods
    members["inherited"]["properties"] = inherited_properties
    members["inherited"]["class_methods"] = inherited_class_methods
    members["inherited"]["decorated_methods"] = inherited_decorated_methods
    
    return members


def populate_class(
    class_path: Path,
    api_rst_path: Path,
    private: bool,
    dunders: bool,
    max_depth: int,
    class_options: dict,
    func_options: dict,
    attr_options: dict,
    class_name: str,
    module_complete_name: str,
    spaces: str
) -> str:
    """Reads the content of the module and returns the string for populating the rst.
    
    Parameters
    ----------
    class_path : Path
        The path of the class file or string to populate.
    
    api_rst_path : Path
        The path of the folder in which the rst files must be generated.
    
    private : bool
        Whether private members should be included in the documentation.
    
    dunders : bool
        Whether dunders should be included in the documentation.
        
    max_depth : int
        Maximum depth of toc tree.
        
    class_options : dict
        A dictionary containing the options to customize the look and feel of
        classes in docs.
        
    func_options : dict
        A dictionary containing the options to customize the look and feel of
        functions and methods in docs.
        
    attr_options : dict
        A dictionary containing the options to customize the look and feel of
        attributes in docs.
        
    class_name : str
        The name of the class to populate.
        
    module_complete_name : str
        The complete name of the module from its root.
        
    spaces : str
        The base spaces to add at the start of new strings to handle with
        classes defined inside modules.

    Returns
    -------
    rst_content_to_add : str
        The content to add to the rst for populating the class doc.
    """
    content = ""
    
    dunder_re = re.compile("__\w+__")
    private_re = re.compile("_\w*")
    full_private_re = re.compile("__\w*")
    
    members = get_class_members(class_path, class_name, module_complete_name)
    
    if "__init__" in members["methods"]:
        members["methods"].remove("__init__")
    if "__init__" in members["inherited"]["methods"]:
        members["inherited"]["methods"].remove("__init__")
    
    # remove dunders or private members if the option is not set
    def values_to_remove(value_list: list[str]) -> list[str]:
        result = []
        for val in value_list:
            if dunder_re.fullmatch(val):
                if not dunders:
                    result.append(val)
            elif private_re.fullmatch(val):
                if not private:
                    result.append(val)
        return result
        
    for key, value in members.items():
        if key != "inherited":
            to_remove = values_to_remove(value)
            members[key] = list(sorted(set(value).difference(to_remove)))
        else:
            for subkey, subvalue in members[key].items():
                to_remove = values_to_remove(subvalue)
                members[key][subkey] = list(sorted(set(subvalue).difference(to_remove)))
    
    def add_entries(
        objects_names: list[str],
        populate_func: Callable[[str, dict], str],
        rubric: str,
        autodoc_directive: str,
        options: dict
    ) -> None:
        nonlocal content, spaces
        if len(objects_names) > 0:
            content += spaces + "   .. rubric:: " + rubric + "\n\n"
            for name in objects_names:
                if (not dunder_re.fullmatch(name) and
                        full_private_re.fullmatch(name) and
                        autodoc_directive == "automethod"):
                    elem_name = "_" + class_name + name
                else:
                    elem_name = name
                content += spaces + "   .. " + autodoc_directive + ":: " + elem_name + "\n"
                content += populate_func(spaces + "   ", options)
            content += "\n"

    def summary_section(rubric_name: str, list_of_elements: list[str]) -> None:
        nonlocal content
        if len(list_of_elements) > 0:
            content += "\n   .. rubric:: " + rubric_name + "\n\n"
            content += "   .. autosummary::\n\n"
            for elem in sorted(list_of_elements):
                if not dunder_re.fullmatch(elem) and full_private_re.fullmatch(elem):
                    elem_name = "_" + class_name + elem
                else:
                    elem_name = elem
                content += "      ~" + class_name + "." + elem_name + "\n"
    
    if class_options["show_inheritance"]:
        content += spaces + "   :show-inheritance:\n"

    content += "\n"
    
    add_entries(members["attributes"], populate_attribute, "Attributes", "autoattribute", attr_options)
    if class_options["show_inherited_members"]:
        add_entries(members["inherited"]["attributes"], populate_attribute, "Inherited attributes", "autoattribute", attr_options)
        

    summary_section("List of properties", members["properties"])
    summary_section("List of methods", members["methods"])
    summary_section("List of decorated methods", members["decorated_methods"])
    summary_section("List of class methods", members["class_methods"])
    
    if class_options["show_inherited_members"]:
        summary_section("List of inherited properties", members["inherited"]["properties"])
        summary_section("List of inherited methods", members["inherited"]["methods"])
        summary_section("List of inherited decorated methods", members["inherited"]["decorated_methods"])

    content += "\n"
        
    def add_inherited():
        if class_options["show_inherited_members"]:
            add_entries(members["inherited"]["properties"], populate_function, "Inherited properties", "autoproperty", func_options)
            add_entries(members["inherited"]["methods"], populate_function, "Inherited methods", "automethod", func_options)
            add_entries(members["inherited"]["decorated_methods"], populate_function, "Inherited decorated methods", "automethod", func_options)
        
    if class_options["inherited_members_before"]:
        add_inherited()
        
    add_entries(members["properties"], populate_function, "Properties", "autoproperty", func_options)
    add_entries(members["methods"], populate_function, "Methods", "automethod", func_options)
    add_entries(members["decorated_methods"], populate_function, "Decorated methods", "automethod", func_options)
    add_entries(members["class_methods"], populate_function, "Class methods", "automethod", func_options)
        
    if not class_options["inherited_members_before"]:
        add_inherited()
    
    return content


def populate_module(
    module_path: Path,
    api_rst_path: Path,
    private: bool,
    dunders: bool,
    max_depth: int,
    pkg_options: dict,
    class_options: dict,
    func_options: dict,
    attr_options: dict,
    module_complete_name: str
) -> str:
    """Reads the content of the module and returns the string for populating the rst.
    
    Parameters
    ----------
    module_path : Path
        The path of the module file to populate.
    
    api_rst_path : Path
        The path of the folder in which the rst files must be generated.
    
    private : bool
        Whether private members should be included in the documentation.
    
    dunders : bool
        Whether dunders should be included in the documentation.
        
    max_depth : int
        Maximum depth of toc tree.
        
    pkg_options : dict
        A dictionary containing the options to customize the look and feel of
        modules in docs.
        
    class_options : dict
        A dictionary containing the options to customize the look and feel of
        classes in docs.
        
    func_options : dict
        A dictionary containing the options to customize the look and feel of
        functions and methods in docs.
        
    attr_options : dict
        A dictionary containing the options to customize the look and feel of
        attributes in docs.
        
    module_complete_name : str
        The complete name of the module from its root.

    Returns
    -------
    rst_content_to_add : str
        The content to add to the rst for populating the module doc.
    """
    separate_content = pkg_options["mod_separate"]
    
    dunder_re = re.compile("__\w+__")
    
    with open(module_path, encoding="utf-8") as f:
        content = f.read()

    def public_member(x): return not x.startswith("_")

    # parse content with abstract syntax trees to get functions, variables,
    # and classes
    variables = []
    functions = []
    classes = []
    module_tree = ast.parse(content)
    for child in ast.iter_child_nodes(module_tree):
        if isinstance(child, (ast.Assign, ast.AugAssign)):
            targets = child.targets if isinstance(child, ast.Assign) else [
                child.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    variables.append(target.id)
    
        if isinstance(child, ast.FunctionDef):
            functions.append(child.name)
    
        if isinstance(child, ast.ClassDef):
            classes.append(child.name)
    # get dunders
    dunders_attr = list(filter(dunder_re.match, variables))
    variables = list(sorted(set(variables).difference(dunders_attr)))
    
    # filter the private members
    if not private:
        variables = list(filter(public_member, set(variables).difference(dunders_attr)))
        functions = list(filter(public_member, functions))
        classes = list(filter(public_member, classes))

    module_content = "\n\n"
    
    def add_entries(
        objects_names: list[str],
        populate_func: Callable[[str, dict], str],
        rubric: str,
        autodoc_directive: str,
        options: dict
    ) -> None:
        nonlocal module_content
        if len(objects_names) > 0:
            module_content += "   .. rubric:: " + rubric + "\n\n"
            for name in sorted(objects_names):
                module_content += "   .. " + autodoc_directive + ":: " + name + "\n"
                module_content += populate_func("   ", options)
            module_content += "\n"

    attributes = variables
    if separate_content:
        module_folder = api_rst_path / ("module_" + module_path.stem)
        
        def starting_content(elem_name: str, autodoc: str) -> str:
            if "__init__" in module_complete_name:
                module_to_write = module_complete_name.rsplit(".", maxsplit=1)[0]
            else:
                module_to_write = module_complete_name

            if elem_name.startswith("_"):
                title = "\\" + elem_name
            else:
                title = elem_name
            
            if autodoc == "autoattribute":
                autodoc_name = module_complete_name + "." + elem_name
            else:
                autodoc_name = elem_name
            
            start_content = title + "\n"
            start_content += "=" * len(title) + "\n\n"
            start_content += ".. currentmodule:: " + module_to_write + "\n\n"
            start_content += ".. " + autodoc + ":: " + autodoc_name + "\n"
            return start_content
        
        def summary_section(
            rubric_name: str,
            file_prefix: str,
            list_of_elements: list[str],
            autodoc_name: str,
            func_to_call: Callable,
            options_to_pass: dict
        ) -> None:
            nonlocal module_content
            module_content += "\n   .. rubric:: " + rubric_name + "\n\n"
            module_content += "   .. autosummary::\n\n"
    
            for elem in sorted(list_of_elements):
                filename = file_prefix + elem + ".rst"
                child_file = module_folder / filename
                module_folder.mkdir(parents=True, exist_ok=True)
                child_file.touch(exist_ok=True)
                child_content = starting_content(elem, autodoc_name)
                if func_to_call != populate_class:
                    child_content += func_to_call("", options_to_pass)
                else:
                    child_content += populate_class(module_path,
                                                    module_folder,
                                                    private,
                                                    dunders,
                                                    max_depth,
                                                    class_options,
                                                    func_options,
                                                    attr_options,
                                                    elem,
                                                    module_complete_name,
                                                    "")
        
                module_content += "      ~" + module_complete_name + "." + elem + "\n"
        
                with open(child_file, "w", encoding="utf-8") as f:
                    f.write(child_content)
    
            module_content += "\n"
            module_content += "   .. toctree::\n"
            module_content += "      :hidden:\n"
            module_content += "      :maxdepth: " + str(max_depth) + "\n\n"
    
            for elem in sorted(list_of_elements):
                filename = file_prefix + elem + ".rst"
                module_content += "      " + module_folder.stem + "/" + filename + "\n"
    
            module_content += "\n"

        if len(attributes) > 0:
            summary_section("Attributes",
                            "attr_",
                            attributes,
                            "autoattribute",
                            populate_attribute,
                            attr_options)
                    
        if len(functions) > 0:
            summary_section("Functions",
                            "func_",
                            functions,
                            "autofunction",
                            populate_function,
                            func_options)
                    
        if len(classes) > 0:
            summary_section("Classes",
                            "class_",
                            classes,
                            "autoclass",
                            populate_class,
                            dict())
    else:
        add_entries(attributes, populate_attribute, "Attributes", "autodata", attr_options)
        add_entries(functions, populate_function, "Functions", "autofunction", func_options)
        
        if len(classes) > 0:
            module_content += "   .. rubric:: Classes\n\n"
            for klass in sorted(classes):
                module_content += "   .. autoclass:: " + klass + "\n"
                module_content += populate_class(module_path,
                                                 api_rst_path,
                                                 private,
                                                 dunders,
                                                 max_depth,
                                                 class_options,
                                                 func_options,
                                                 attr_options,
                                                 klass,
                                                 module_complete_name,
                                                 "   ")
    
    module_content += "\n"
    
    return module_content


def generate_package_rst_files(
    structure: dict,
    package_path: Path,
    api_package_path: Path,
    private: bool,
    dunders: bool,
    max_depth: int,
    pkg_options: dict,
    class_options: dict,
    func_options: dict,
    attr_options: dict,
    parent_name: str = ""
) -> None:
    """Generate rst files for the documentation.
    
    Parameters
    ----------
    structure : dict
        The structure of the package.
    
    package_path : Path
        The path of the package.
    
    api_package_path : Path
        The path of the folder in which the rst files must be generated.
    
    private : bool
        Whether private members should be included in the documentation.
    
    dunders : bool
        Whether dunders should be included in the documentation.
        
    max_depth : int
        Maximum depth of toc tree.
        
    pkg_options : dict
        A dictionary containing the options to customize the look and feel of
        modules in docs.
        
    class_options : dict
        A dictionary containing the options to customize the look and feel of
        classes in docs.
        
    func_options : dict
        A dictionary containing the options to customize the look and feel of
        functions and methods in docs.
        
    attr_options : dict
        A dictionary containing the options to customize the look and feel of
        attributes in docs.
        
    parent_name : str, default=""
        The full name of the parent package.

    Returns
    -------
    None
    """
    complete_pkg_title = pkg_options["complete_pkg_title"]
    
    # generate the index.rst of the package
    index_rst = api_package_path / "index.rst"
    index_rst.parent.mkdir(parents=True, exist_ok=True)
    index_rst.touch()
    
    content = ""
    
    # insert the content of the init module
    package_name = list(structure.keys())[0]
    if parent_name == "":
        package_complete_name = package_name
    else:
        package_complete_name = parent_name + "." + package_name
    if complete_pkg_title:
        title = package_complete_name + " package"
    else:
        title = package_name + " package"
    content += title + "\n" + "=" * len(title) + "\n\n"
    content += ".. currentmodule:: " + package_complete_name + "\n\n"
    content += ".. automodule:: " + package_complete_name + "\n"
    module_complete_name = package_complete_name + ".__init__"
    content += populate_module(package_path / "__init__.py",
                               api_package_path,
                               private,
                               dunders,
                               max_depth,
                               pkg_options,
                               class_options,
                               func_options,
                               attr_options,
                               module_complete_name) + "\n\n"
    
    # insert the list of pure subpackages
    if len(structure[package_name][PACKAGES_KEY]) != 0:
        content += "Subpackages\n-----------\n\n.. toctree::\n"
        content += "   :maxdepth: " + str(max_depth) + "\n\n"
        for pkg in structure[package_name][PACKAGES_KEY]:
            content += "   " + list(pkg.keys())[0] + "/index\n"
        content += "\n"
    
    # insert the list of submodules
    if len(structure[package_name][MODULE_KEY]) > 1:
        content += "Submodules\n----------\n\n.. toctree::\n"
        content += "   :maxdepth: " + str(max_depth) + "\n\n"
        for mod in structure[package_name][MODULE_KEY]:
            if mod != "__init__":
                content += "   " + mod + "\n"
        content += "\n"

    with open(index_rst, "w", encoding="utf-8") as f:
        f.write(content)
        
    # create all submodules rst files
    for mod in set(structure[package_name][MODULE_KEY]).difference(["__init__"]):
        content = ""
        module_rst = api_package_path / f"{mod}.rst"
        module_rst.touch()
        module_complete_name = package_complete_name + "." + mod
        if complete_pkg_title:
            title = module_complete_name + " module"
        else:
            title = mod + " module"
        content += title + "\n" + "=" * len(title) + "\n\n"
        content += ".. currentmodule:: " + module_complete_name + "\n\n"
        content += ".. automodule:: " + module_complete_name + "\n"
        content += populate_module(package_path / f"{mod}.py",
                                   api_package_path,
                                   private,
                                   dunders,
                                   max_depth,
                                   pkg_options,
                                   class_options,
                                   func_options,
                                   attr_options,
                                   module_complete_name) + "\n\n"
        
        with open(module_rst, "w", encoding="utf-8") as f:
            f.write(content)
            
    # pass the call to subpackages to build them
    for pkg in structure[package_name][PACKAGES_KEY]:
        pkg_name = list(pkg.keys())[0]
        generate_package_rst_files(pkg,
                                   package_path / pkg_name,
                                   api_package_path / pkg_name,
                                   private,
                                   dunders,
                                   max_depth,
                                   pkg_options,
                                   class_options,
                                   func_options,
                                   attr_options,
                                   package_complete_name)


if __name__ == "__main__":
    options = process_arguments(sys.argv)
    
    package_path = Path(os.getcwd()) / options["package_location"]
    package = get_package_structure(package_path, options["private_packages_modules"])
    
    given_source = Path(options["source"])
    given_api = Path(options["api_location"])
    source_folder = given_source if given_source.is_absolute() else Path(os.getcwd()) / given_source
    api_folder = source_folder / options["api_location"]
    
    options_packages = {"complete_pkg_title": options["complete_package_title"],
                        "mod_separate": options["mod_separate"]}
    options_classes = {"show_inheritance": options["show_inheritance"],
                       "show_inherited_members": options["show_inherited_members"] or options["inherited_members_before"],
                       "inherited_members_before": options["inherited_members_before"]}
    options_functions = dict()
    options_attributes = {"show_value": options["show_attribute_value"]}
    
    generate_package_rst_files(package,
                               package_path,
                               api_folder / package_path.name,
                               options["private_members"],
                               options["dunders"],
                               options["max_depth"],
                               options_packages,
                               options_classes,
                               options_functions,
                               options_attributes)
