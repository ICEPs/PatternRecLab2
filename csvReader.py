import csv

list = [] #initializing the list that will hold all values
with open('csvfile.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        #print row
		newRow = [int(n) for n in row]; #turning values into integers
		list.append(newRow);
		
sortedList = sorted(list,key=lambda l:l[64], reverse=True) #sorting the list by the 64th row, which classifies whether something is an object or not
for row in sortedList:
	print row[64]
	#hi EJ! You can just change this however you like :)
	#basta what this does is simply go through the sorted list :)