import re

old_file = 'deu.txt'
new_file = 'new_deu.txt'

f1 = open(old_file, 'r', encoding='utf-8')
f2 = open(new_file, 'w', encoding='utf-8')

s = f1.readline()
while s:
	if re.search(r'\d', s)==None:
		f2.write(s)
	s = f1.readline()
f1.close()
f2.close()