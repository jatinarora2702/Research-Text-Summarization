import json
import sys
from collections import OrderedDict

filename = sys.argv[1]


papers=[]


basepath = "../../../Test_Data/Output_Dict/"


def get_paras(filename):
	train=open(basepath+filename+'.txt','r+')

	print("in get para")
	for line in train:

		line = line.replace("###FORMULA###","||FORMULA||")
		line = line.replace("###TABLE###","||TABLE||")
		line = line.replace("###FIGURE###","||FIGURE||")

		
		map=line.decode('utf-8').split('\t')
		paper=dict()
		paper['id']=map[0]
		paper['name']=map[1]
		try:
			paper['info']=json.loads(map[2],object_pairs_hook=OrderedDict)
		except:
			continue
		paper['sum']=map[3]
		if (len(paper['sum'])>=10):
			papers.append(paper)
	print("Papers: ", len(papers))
	for paper in papers:
		print paper['id']
		paper['sum']=paper['sum'].encode('utf-8')
		paper_data=""
		for key in paper['info']:
			if key.lower().strip().startswith("introduction"):
				for item in paper['info'][key]:
					if isinstance(item,unicode):
						paper_data+=item+" "
					elif isinstance(item,str):
						paper_data+=item+" "
					elif isinstance(item,dict):
						for innerKey in item:
							for innerItem in item[innerKey]:
								if (isinstance(innerItem,unicode)):
									paper_data+=innerItem+" "
								elif (isinstance(innerItem,str)):
									paper_data+=innerItem+" "
								elif isinstance(innerItem,dict):
									for in_innerKey in innerItem:
										for in_innerItem in innerItem[in_innerKey]:
											if (isinstance(in_innerItem,unicode)):
												paper_data+=in_innerItem+" "
											elif (isinstance(in_innerItem,str)):
												paper_data+=in_innerItem+" "

		paper_data=paper_data.encode('utf-8')
		para_list = paper_data.split("\n\n")

		for idx, para_elem in enumerate(para_list):
			para_list[idx]=para_list[idx].replace("\n"," ")
		print("End ", paper['sum'])
		return (para_list, paper['sum'])


