import os
import shutil
path = 'output'

for root, dirs, files in os.walk(path):
    print(root)
    if root == path:
        print('1')
        continue
    clean_flag = True
    for f in files:
        if f.endswith('.png'):
            clean_flag = False
    if clean_flag:
        print(f"{root} need clean")
        print(f"{root} has been deletedd")
        shutil.rmtree(root)
