#!/usr/bin/env python3

# not working on GPFS
import struct, fcntl, os, time
import subprocess
# 
# with open("lockfile","a") as f:
#     fcntl.lockf(f, fcntl.LOCK_EX)
#     print("sleep")
#     time.sleep(30)
#     print("finished")


while True:
    try:
        with open("lockfile","x") as f:
            print("ok")
        subprocess.run(["rm","lockfile"])
        break
    except FileExistsError:
        print("sleep")
        time.sleep(1)

print("done")
