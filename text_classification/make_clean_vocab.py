vocab_file = 'vocab.txt'
clean_vocab_file = 'clean_vocab.txt'
vocab = {}

f = open(vocab_file, 'r')
s = f.readline()
while s:
	items = s.split(' ')
	if items[0].isalpha():
		vocab[items[0]] = int(items[1])
	s = f.readline()
f.close()

print(len(vocab))

f = open(clean_vocab_file, 'w')
for item in vocab:
	f.write(item + ' ' + str(vocab[item]) + '\n')
f.close()