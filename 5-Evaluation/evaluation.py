import sys
import os
from pythonrouge import pythonrouge
import plotly.plotly as py
import plotly.graph_objs as go
import plotly

base_dir = sys.argv[1]
method = sys.argv[2]

ROUGE_PATH = base_dir + '/evaluation/pythonrouge/RELEASE-1.5.5/ROUGE-1.5.5.pl'
DATA_PATH = base_dir + '/evaluation/pythonrouge/RELEASE-1.5.5/data'

r_1 = 0
r_2 = 0
r_3 = 0
r_l = 0
r_su_4 = 0
count = 0

num = []
for x in range(3, len(sys.argv)):
	num.append(sys.argv[x])

plot_r_1 = []
plot_r_2 = []
plot_r_3 = []
plot_r_l = []
plot_r_su_4 = []

summary_dir = base_dir + "/data-3000/summary/" + method + "/"
model_dir = base_dir + "/data-3000/model/"

def read_from_file(file_name):
	f = open(file_name, 'r')
	data = f.read()
	return data[:-1]


for n in num:
	summary_n_dir = summary_dir + str(n) + "/"
	summary_files = os.listdir(summary_n_dir)

	for summary_file in summary_files:
		summary_file_path = summary_n_dir + summary_file;
		model_file_path = model_dir + summary_file;

		summary = read_from_file(summary_file_path)
		model = read_from_file(model_file_path)

		score = pythonrouge.pythonrouge(model, summary, ROUGE_PATH, DATA_PATH)

		r_1 += score['ROUGE-1']
		r_2 += score['ROUGE-2']
		r_3 += score['ROUGE-3']
		r_l += score['ROUGE-L']
		r_su_4 += score['ROUGE-SU-4']
		count += 1

	r_1 /= count
	r_2 /= count
	r_3 /= count
	r_l /= count
	r_su_4 /= count
	
	plot_r_1.append(r_1)
	plot_r_2.append(r_2)
	plot_r_3.append(r_3)
	plot_r_l.append(r_l)
	plot_r_su_4.append(r_su_4)


trace1 = go.Scatter(
    x = num,
    y = plot_r_1,
    mode = 'lines',
    name = 'Rouge-1'
)

trace2 = go.Scatter(
    x = num,
    y = plot_r_2,
    mode = 'lines',
    name = 'Rouge-2'
)

trace3 = go.Scatter(
    x = num,
    y = plot_r_3,
    mode = 'lines',
    name = 'Rouge-3'
)

tracel = go.Scatter(
    x = num,
    y = plot_r_l,
    mode = 'lines',
    name = 'Rouge-L'
)

trace_su_4 = go.Scatter(
    x = num,
    y = plot_r_su_4,
    mode = 'lines',
    name = 'Rouge-SU-4'
)

data = [trace1, trace2, trace3, tracel, trace_su_4]

fig = plotly.offline.plot({
"data": data,
"layout": plotly.graph_objs.Layout(showlegend=True,
    height=600,
    width=800,
    xaxis = dict(title = 'No. of Sentance'),
    yaxis = dict(title = 'Accuracy'),
    title = "Method " + method
)
}, 
filename = base_dir + "/graph/" + method + ".html"
)
