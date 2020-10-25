import datetime
import numpy as np


'''
a = []
for i in range(500):
	a.append(np.zeros([128, 160, 3], dtype=np.float32))

start1 = datetime.datetime.now()	
for i in range(100):
	b = []
	for j in range(50):
		k = np.random.randint(low=0, high=500-4+1)
		sb = []
		for m in range(k, k+4):
			sb.append(a[m])
		sb = np.concatenate(sb, axis=2)
		b.append(sb)
	b = np.float32(b)

end1 = datetime.datetime.now()

print('Delta time 1', end1 - start1)
'''

a = []
for i in range(500):
	a.append(np.random.rand(128, 160, 3))
a = np.float32(a)
b = np.zeros([50,4,128,160,3],dtype=np.float32)
start2 = datetime.datetime.now()
for i in range(100):
	for j in range(50):
		k = np.random.randint(low=0, high=500-4+1)
		b[j] = a[k:k+4]
end2 = datetime.datetime.now()
print('Delta time 2', end2 - start2)
