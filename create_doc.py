import os
import platform

if __name__ == "__main__":
    if platform.system() == "Windows":
        os.system("powershell rm -r ./docs/source/reference/api")
        os.system("powershell rm -r ./docs/source/developer/api")
    else:
        os.system("rm -r ./docs/source/reference/api")
        os.system("rm -r ./docs/source/developer/api")
    
    os.system("poetry run python ./doc_creator.py --max-depth 2 -d -i "
              "--show-inherited-members --mod-separate anomalearn")
    os.system("poetry run python ./doc_creator.py -a ./developer/api "
              "--max-depth 2 -p -P -d -i --show-inherited-members "
              "--mod-separate anomalearn")
    os.system("poetry run sphinx-build -b html ./docs/source ./docs/stable")
    os.system("poetry run python ./doc_fix_dev_api.py ./docs/stable")
