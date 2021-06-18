#import libraries
import csv

filename = 'breast_cancer_data.csv'

training =[]
testing = []

count0 = 0
count1 = 0

with open(filename,'r+') as file:
	csv_reader = csv.reader(file)
	for row in csv_reader:
		if row[-1] == '0.0':
			count0 += 1
		else:
			count1 +=1
		if (count0 % 10) == 0 or (count1 % 10) == 0:
			testing.append(row)
		else:
			training.append(row)

with open('breast_cancer_training.csv','w+',newline='') as file:
	csvWrite = csv.writer(file)
	csvWrite.writerows(training)

with open('breast_cancer_testing.csv','w+',newline='') as file:
	csvWrite = csv.writer(file)
	csvWrite.writerows(testing)