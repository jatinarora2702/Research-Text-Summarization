from __future__ import print_function
import codecs
import os
import json
import collections
import re
import sys


math = sys.argv[1]

basepath_orig = "/home/du3/13CS30043/SNLP/Dataset/Papers_Text_New_Cat/"

destpath_new = "/home/du3/13CS30043/SNLP/Dataset/Papers_Text_Without_Eq_Cat/"

basepath = "/home/du3/13CS30043/SNLP/Dataset/Papers_Text_Without_Eq_Cat/"
destpath = "/home/du3/13CS30043/SNLP/Dataset/all-parsed-papers-category.txt"


symbol_path = os.path.join(os.path.dirname(__file__),"Symbol_Files/symbols.txt")

symbol_1_path = os.path.join(os.path.dirname(__file__),"Symbol_Files/symbols_1.txt")

binary_path = os.path.join(os.path.dirname(__file__),"Symbol_Files/binary.txt")


def remove_math():
	symbols = []

	binary = []

	symbol_file = codecs.open(symbol_path, 'r', 'utf-8')

	for row in symbol_file:
		s = row.strip().split('\t')
		symbols.append(s[0].strip())
		symbols.append(s[1].strip())

	symbol_1_file = codecs.open(symbol_1_path, 'r', 'utf-8')

	for row in symbol_1_file:
		s = row.strip().split('\t')
		symbols.append(s[0].strip())

	binary_file = codecs.open(binary_path, 'r', 'utf-8')

	for row in binary_file:
		s = row.strip().split('\t')
		binary.append(s[0].strip())


	files = os.listdir(basepath_orig)

	symbol_string = ""

	for symbol in symbols:
		symbol_string += symbol

	binary_string = ""
	for bin in binary:
		symbol_string += bin

		binary_string += bin


	print("[^ ]*["+symbol_string+"][^ ]*")

	regex_symbol = re.compile(ur" [^ @#]*["+symbol_string+"][^ ]* ", re.U)

	regex_binary= re.compile(ur" [^ @#]* ["+binary_string+"] [^ ]* ", re.U)

	regex_min = re.compile(" [^ ]*[_][^ ]* ", re.U)

	regex_1 = re.compile(" [a-zA-Z]+[ ]?[-/][ ]?[a-zA-Z]+ ", re.U)

	regex_2 = re.compile(" [^a-zA-Z ]+[ ]?[-/][ ]?[^a-zA-Z ]+ ", re.U)


	for filename in files:
		file = codecs.open(basepath_orig+filename,'r','utf-8')

		destfile = codecs.open(destpath_new+filename, 'w', 'utf-8')

		text = file.read()

		cnt = 0

		for m in regex_symbol.finditer(text):
			text = text[:m.start()-cnt]+' ||SYMBOLTOKEN|| '+text[m.end()-cnt:]
			cnt = cnt+(m.end()-m.start())-len(" ||SYMBOLTOKEN|| ")

		cnt = 0
		for m in regex_binary.finditer(text):
			text = text[:m.start()-cnt]+' ||MATHEQUATION|| '+text[m.end()-cnt:]
			cnt = cnt+(m.end()-m.start())-len(" ||MATHEQUATION|| ")


		cnt = 0
		for m in regex_1.finditer(text):
			text = text[:m.start()-cnt]+' '+text[m.start()-cnt+1:m.end()-cnt].replace('-','')+text[m.end()-cnt:]
			cnt = cnt+(m.end()-m.start())-1

		cnt = 0
		for m in regex_2.finditer(text):
			text = text[:m.start()-cnt]+' ||MATHEQUATION|| '+text[m.end()-cnt:]
			cnt = cnt+(m.end()-m.start())-len(" ||MATHEQUATION|| ")


		cnt = 0
		for m in regex_min.finditer(text):
			text = text[:m.start()-cnt]+' ||SYMBOLTOKEN|| '+text[m.end()-cnt:]
			cnt = cnt+(m.end()-m.start())-len(" ||SYMBOLTOKEN|| ")


		print(text, file = destfile)
		print(filename)





def main():

	if math == "1":
		remove_math()

	
	files = os.listdir(basepath)

	destfile = codecs.open(destpath,'w','utf-8')

	err_cnt = 0
	ii = 0

	for filename in files:
		try:
			file = codecs.open(basepath+filename,'r','utf-8')

			title_f = False

			section_f = False

			sub_section_f = False

			sub_sub_section_f = False

			abstract_f = False

			formula_f = False

			table_f = False
			figure_f = False

			title = ""
			abstract = ""


			section_head = ""
			sub_section_head = ""
			sub_sub_section_head = ""

			section_text = ""
			sub_section_text = ""
			sub_sub_section_text = ""


			section_s = False
			sub_section_s = False
			sub_sub_section_s = False

			dict_file = collections.OrderedDict()

			for row in file:

				if row.strip() == "@#^T":
					title_f = False
					continue
				elif row.strip() == "@#^A":
					abstract_f = False
					continue

				elif row.strip() == "@@@FORMULA@@@":
					formula_f = False
					continue

				elif row.strip() == "@@@TABLE@@@":
					table_f = False
					continue

				elif row.strip() == "@@@FIGURE@@@":
					figure_f = False
					continue

				if table_f == True:
					continue

				if formula_f == True:
					continue

				if figure_f == True:
					continue


				elif row.strip() == "@#^S":
					section_f = False
					section_s = True

					if section_head == '':
						section_head = "DEFAULT"

					if section_head not in dict_file.keys():
						dict_file[section_head] = ["",collections.OrderedDict()]
					continue

				elif row.strip() == "@#S^S":
					sub_section_f = False
					sub_section_s = True

					if sub_section_head == '':
						sub_section_head = 'DEFAULT'


					if sub_section_head not in dict_file[section_head][1].keys():
						dict_file[section_head][1][sub_section_head] = ["",collections.OrderedDict()]
					continue

				elif row.strip() == "@S#S^S":

					if sub_sub_section_head == '':
						sub_sub_section_head = 'DEFAULT'

					if sub_section_head.strip() in dict_file[section_head][1]:

						sub_sub_section_f = False
						sub_sub_section_s = True

						# print(filename,section_head,sub_section_head,sub_sub_section_head)

						if sub_sub_section_head not in dict_file[section_head][1][sub_section_head][1].keys():
							dict_file[section_head][1][sub_section_head][1][sub_sub_section_head] = ["",collections.OrderedDict()]
					else:
						sub_section_f = False
						sub_section_s = True

						if sub_section_head == '':
							sub_section_head = 'DEFAULT'


						if sub_section_head.strip() not in dict_file[section_head][1].keys():
							dict_file[section_head][1][sub_section_head] = ["",collections.OrderedDict()]
						
						if sub_sub_section_head not in dict_file[section_head][1][sub_section_head][1].keys():
							dict_file[section_head][1][sub_section_head][1][sub_sub_section_head] = ["",collections.OrderedDict()]

					continue

				if title_f == True:
					title = title+row.strip()+" "

				elif abstract_f == True:
					abstract = abstract+row.strip()+" "

				elif section_f == True:
					section_head = section_head+row.strip()

				elif section_f == True:
					section_head = section_head+row.strip()

				elif sub_section_f == True:
					sub_section_head = sub_section_head+row.strip()

				elif sub_sub_section_f == True:
					sub_sub_section_head = sub_sub_section_head+row.strip()





				if row.strip() == "@#!T":
					title_f = True
				elif row.strip() == "@#!A":
					abstract_f = True
				elif row.strip() == "@#!S":
					section_f = True
					section_head = ""
					sub_sub_section_s= False
					sub_section_s = False
					section_s = False

					
				elif row.strip() == "###FORMULA###":
					row = "||FORMULA||"
					formula_f = True
				elif row.strip() == "###TABLE###":
					row = "||TABLE||"
					table_f = True

				elif row.strip() == "###FIGURE###":
					row = "||FIGURE||"
					figure_f = True


					

				elif row.strip() == "@#S!S":
					sub_section_f = True

					sub_section_head = ""
					sub_sub_section_s= False
					sub_section_s = False
					section_s = False


				elif row.strip() == "@S#S!S":

				
					sub_sub_section_f = True

					sub_sub_section_head = ""
					sub_sub_section_s= False
					section_s = False
					sub_section_s = False


				if section_s == True:
					dict_file[section_head][0] = dict_file[section_head][0]+row

				elif sub_section_s == True:
					dict_file[section_head][1][sub_section_head][0] = dict_file[section_head][1][sub_section_head][0]+row

				elif sub_sub_section_s == True:
					dict_file[section_head][1][sub_section_head][1][sub_sub_section_head][0] = dict_file[section_head][1][sub_section_head][1][sub_sub_section_head][0]+row
			print(filename)

			if dict_file != collections.OrderedDict():
				print(str(filename)+"\t"+str(title.strip())+'\t'+json.dumps(dict_file)+'\t'+abstract.strip(), file= destfile)

				ii = ii+1

				# if ii == 100:
				# 	break

		except:
			print("Error",filename)
			err_cnt += 1


	print("Error Count ", err_cnt)
	print("Total Count ", ii)





if __name__ == "__main__":main()
