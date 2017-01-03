import os
import sys 

base_dir = sys.argv[1]
method = sys.argv[2]
num = sys.argv[3]
ip_dir = base_dir + "/data-3000/input/"
op_dir = base_dir + "/data-3000/summary/"

ip_files = os.listdir(ip_dir)

for ip_file in ip_files:
	ip_file_path = ip_dir + ip_file
	op_file_path = op_dir + method + "/" + num
	if not os.path.exists(op_file_path):
		os.makedirs(op_file_path)
	op_file_path = op_file_path+ "/" + ip_file
	
	command = "sumy " + method + " --length=" + num + " --file=\'" + ip_file_path + "\' > \'" + op_file_path + "\'"
	os.system(command)
	print(ip_file)

print("Method : " + method + ", Num : " + num)
