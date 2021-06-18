#import library
import csv

filename = 'breast_cancer_data_old.csv'

data=[]
with open(filename, 'r+') as file:
	csv_reader = csv.reader(file)
	for row in csv_reader:
		if not row:
			continue
		row.pop(0)
		row.insert(len(row)-1, row.pop(0))
		if row[-1]=='M':
			row[-1]='1.0'
		else:
			row[-1]='0.0'
		data.append(row)

with open('breast_cancer_data.csv','w+', newline='') as file:
	csvWrite = csv.writer(file)
	csvWrite.writerows(data)