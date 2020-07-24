import os

curr_dir = os.fsencode(os.getcwd())
for file_obj in os.listdir(curr_dir):
    filename = os.fsdecode(file_obj)
    if 'ents' in filename:
        with open(filename, 'r+') as f:
            content = f.read()
            f.seek(0)
            f.truncate()
            f.write(content.replace('G', ' '))
