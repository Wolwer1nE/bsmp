import random
import sys

with open(sys.argv[2], "w") as out:
    out.writelines(str(random.uniform(int(sys.argv[3]), int(sys.argv[4]))) + "\n" for i in range(int(sys.argv[1])))
