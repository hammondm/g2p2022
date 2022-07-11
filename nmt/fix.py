#fix src files

import sys

filename = sys.argv[1]

f = open(filename,'r')
t = f.read()
f.close()

t = t.split('\n')

for line in t:
	res = ' '.join(
		[letter for letter in line]
	)
	if len(res) > 0: print(res)

