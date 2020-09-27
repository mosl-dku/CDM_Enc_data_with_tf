f = open('dataset_age_G_H_W_D.csv','r')

header = ''
line_cnt = 0
data_cnt = 0
dataset =[]
divide_str=''
is_first = 1
while True:
	line = f.readline()
	line_cnt+=1
	if not line:
		divide_str = divide_str[:-1]
		dataset.append(divide_str)
		print('file -end')
		break
	
	if is_first ==1:
		header = line
		is_first = 0

	if divide_str == '':
		divide_str=divide_str+header+','
		continue
	if line_cnt != 40000:
		divide_str = divide_str + line + ','
	elif line_cnt == 40000:
		divide_str +=line
		dataset.append(divide_str)
		divide_str = ''
		line_cnt =0
		
#print(len(dataset))
#print(dataset[0])
f.close()

for x in range(len(dataset)):
	file_name = 'divide_dataset_2/data' + str(x) +'.txt'
	mini = open(file_name,'w')
	mini.write(dataset[x])
	mini.close()
