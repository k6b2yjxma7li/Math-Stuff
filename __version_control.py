"""
Takes file {name}.{ext}
Checks if:
    + version line exists
    + version line is up-to-date
If one or more of above is false:
    + creates file {name}.tmp
    + writes each line from {name}.{ext} to {name}.tmp and adds version line
    + changes extension of {name}.{ext} to {name}.tmp and vice versa
    + deletes new tmp files when all files have been switched

version: 20190823.1
"""

# This file will change the versions of all Python files in project
import os
import glob
from datetime import date
import re

today_date = str(date.today()).split("-")
version = "".join(today_date)

version_pattern = "^(version: )(\d{4}[0-1]\d[0-3]\d)(\.)(\d+)$"
for docline in __doc__:
    check = re.match(version_pattern, docline)
    if check is not None:
        if re.match(version, docline) is not None:
            version_nr = int(check.split('.')[-1])
        else:
            version_nr = 0
    else:
        version_nr = 0

version = "version: " + version + f".{version_nr+1}"

# file_names = glob.glob("*.py")
file_names = ["__1.py"]
header = ""
for file_name in file_names:
    is_version = False
    if file_name.find(".py") != -1:
        fname, ext = file_name.split(".")
        with open(file_name) as fl:
            docstr_head = False
            docstr_tail = True
            tmp_fl = open(f"{fname}.tmp", 'w')
            for line in fl:
                docstr_head ^= (line.find("\"\"\"") != -1)
                if line.find("\"\"\"") != -1:
                    header = line[0:line.find("\"\"\"")]
                if re.match(version_pattern, line) is not None:
                    print(f"{header}{version}", file=tmp_fl)
                    print(f"{header}{version}")
                    # next(fl)
                else:
                    print(f"{line}", end="")
                    # if not docstr_tail and not docstr_head:
                    #     print(f"{header}{version}\n{header}\"\"\"",
                    #           file=tmp_fl)
                    #     print(f"{header}{version}\n{header}\"\"\"")
                    # else:
                    #     print(line, end="", file=tmp_fl)
                    #     print(line, end="")
                docstr_tail ^= (line.find("\"\"\"") != -1)
            tmp_fl.close()
        # os.system(f"rm {}.")

os.system("rm *.tmp")
