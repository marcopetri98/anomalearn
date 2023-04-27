import argparse
import os
import re
import sys
from pathlib import Path


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
    parser = argparse.ArgumentParser(prog="doc-fix-dev-api.py",
                                     description="Fix wrong links of dev api reference")
    
    parser.add_argument("docs_html_root",
                        metavar="DOCS_ROOT",
                        help="it is the relative location in which the deployed"
                             " documentation can be found, relatively to the "
                             "current working directory.")
    
    output = parser.parse_args(argv[1:])
    arguments = vars(output)
    
    given_docs = Path(arguments["docs_html_root"])
    
    if not given_docs.is_dir():
        raise ValueError("The given docs root is not a directory.")
    
    return arguments


if __name__ == "__main__":
    options = process_arguments(sys.argv)
    
    wrong_links_re = re.compile('href="\.\./[\w\./#]*reference/api/[\w\./#]*"')
    
    docs_root = Path(options["docs_html_root"])
    docs_dev_api = docs_root / "developer/api"
    
    for root, dirs, files in os.walk(docs_dev_api):
        for file in files:
            file_path = Path(root, file)
            
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            
            if wrong_links_re.search(content):
                all_links = wrong_links_re.findall(content)
                new_content = content
                
                for link in all_links:
                    correct_link = link.replace("../reference/api/", "../developer/api/")
                    new_content = new_content.replace(link, correct_link)
                    
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
