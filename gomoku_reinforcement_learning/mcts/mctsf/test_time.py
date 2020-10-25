import datetime
import numpy as np

a = list(range(100))

start = datetime.datetime.now()
for i in range(10000):
	a.append(100)
	a.append(101)
	a.append(102)
	
	a.remove(100)
	a.remove(101)
	a.remove(102)
end = datetime.datetime.now()
print('delta 1', end - start)

a = set(list(range(100)))
b = set(a)
c = set([100, 101, 102])
start = datetime.datetime.now()
for i in range(10000):
	a - b
end = datetime.datetime.now()
print('delta 2', end - start)

a = np.arange(103)
start = datetime.datetime.now()
for i in range(10000):
	y = np.where(a<100)
end = datetime.datetime.now()
print('delta 3', end - start)