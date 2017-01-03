import sys
import os
import random
import glob

def move_files(filelist, source, dest):
	for file in filelist:
		os.rename(file, os.path.join(dest, os.path.basename(file)))

def ensure_exists(name):
	if not os.path.exists(name):
		os.makedirs(name)
	else:
		os.remove(name)
		os.makedirs(name)


data_folder = sys.argv[-2]
dest_folder = sys.argv[-1]


filelist = glob.glob(data_folder)

assert filelist, 'Empty filelist.'

random.shuffle(filelist)

dirname = dest_folder
train_dir = os.path.join(dirname, "train")
val_dir = os.path.join(dirname, "validation")
test_dir = os.path.join(dirname, "test")



cnt = len(filelist)
print(data_folder, dest_folder)
print(filelist)
l = len(filelist)

train_no = int(0.7*l)
validation_no = int(0.15*l)

train_files = filelist[:train_no]


print(len(train_files))

validation = filelist[train_no:train_no+validation_no]

print(len(validation))

test = filelist[train_no+validation_no:]

print(len(test))

dirname = os.path.dirname(filelist[0])

ensure_exists(train_dir)
ensure_exists(val_dir)
ensure_exists(test_dir)

move_files(train_files, dirname, train_dir)
move_files(validation, dirname, val_dir)
move_files(test, dirname, test_dir)

