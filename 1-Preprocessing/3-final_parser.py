import json


papers=[]
train=open('all-parsed-papers-category.txt','r+')

for line in train:

	line = line.replace("###FORMULA###","||FORMULA||")
	line = line.replace("###TABLE###","||TABLE||")
	line = line.replace("###FIGURE###","||FIGURE||")

	
	map=line.decode('utf-8').split('\t')
	paper=dict()
	paper['id']=map[0]
	paper['name']=map[1]
	try:
		paper['info']=json.loads(map[2])
	except:
		continue
	paper['sum']=map[3]
	if (len(paper['sum'])>=10):
		papers.append(paper)

for paper in papers:
	print paper['id']
	paper['sum']=paper['sum'].encode('utf-8')
	paper_data=""
	for key in paper['info']:
		# if key.lower().strip() == "introduction" or key.lower().strip().startswith("conclusion") or key.lower().strip().startswith("future work"):
		# 	continue
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
	paper_data=paper_data.replace("\n"," ")
	data_input=open('./data-cat-all/input/'+paper['id'],'w+')
	data_model=open('./data-cat-all/model/'+paper['id'],'w+')
	print>> data_input,paper_data
	print>> data_model,paper['sum']
