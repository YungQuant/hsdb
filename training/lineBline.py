import os

path = "../../HSDB-BMEX_XBTUSD0_10up.txt"
if os.path.isfile(path) == False:
    print(f'could not source {path} data')
else:
    fileP = open(path, "r")
    lines = fileP.readlines(1000000)
    [print(line, "\n") for line in lines]
        
    
