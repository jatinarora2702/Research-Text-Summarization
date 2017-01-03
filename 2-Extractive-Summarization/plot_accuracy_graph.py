import sys
import os

summary_method = ['lex-rank'];
summary_length = ['50','75','100']

base_dir = os.path.join(os.path.dirname(__file__), '../0-Sample-Data/')

for method in summary_method:
	for num in summary_length:
		command = "python ./base_line/baseline.py \'" + base_dir + "\' " + method + " " + str(num)
		os.system(command)

for method in summary_method:
	command = "python ./evaluation/evaluation.py \'" + base_dir + "\' " + method
	for num in summary_length:
		command = command + " " + str(num)
	os.system(command)

