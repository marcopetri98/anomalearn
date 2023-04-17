import argparse
import os
import sys
from pathlib import Path


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


def populate_module(module_path: Path, private: bool, dunders: bool) -> str:
    """Reads the content of the module and returns the string for populating the rst.
    
    Parameters
    ----------
    module_path : Path
        The path of the module file to populate.
    
    private : bool
        Whether private members should be included in the documentation.
    
    dunders : bool
        Whether dunders should be included in the documentation.

    Returns
    -------
    rst_content_to_add : str
        The content to add to the rst for populating the module doc.
    """
    return ""


def generate_package_rst_files(
    structure: dict,
    package_path: Path,
    api_package_path: Path,
    private: bool,
    dunders: bool,
    max_depth: int,
    complete_pkg_title: bool,
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
        
    complete_pkg_title : bool
        States if the complete python package must be the title.
        
    parent_name : str, default=""
        The full name of the parent package.

    Returns
    -------
    None
    """
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
    content += ".. automodule:: " + package_complete_name
    content += populate_module(package_path / "__init__.py", private, dunders) + "\n\n"
    
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
        content += ".. automodule:: " + module_complete_name
        content += populate_module(package_path / f"{mod}.py", private, dunders) + "\n\n"
        
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
                                   complete_pkg_title,
                                   package_complete_name)


if __name__ == "__main__":
    options = process_arguments(sys.argv)
    
    package_path = Path(os.getcwd()) / options["package_location"]
    package = get_package_structure(package_path, options["private_packages_modules"])
    
    given_source = Path(options["source"])
    given_api = Path(options["api_location"])
    source_folder = given_source if given_source.is_absolute() else Path(os.getcwd()) / given_source
    api_folder = source_folder / options["api_location"]
    
    generate_package_rst_files(package,
                               package_path,
                               api_folder / package_path.name,
                               options["private_members"],
                               options["dunders"],
                               options["max_depth"],
                               options["complete_package_title"])
