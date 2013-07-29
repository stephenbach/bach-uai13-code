#!/usr/bin/python

import re, math, sys

# get command line args
if len(sys.argv) < 2:
	print 'USAGE: python computeSim.py <outFile> [ <thresh: off=0,avg=1,avg+std=2> <rescale: off=0,on=1> ]'
	exit()
outFile = sys.argv[1]
threshold = 0
if len(sys.argv) >= 3:
	flag = int(sys.argv[2])
	if flag in [0,1,2]:
		threshold = flag
rescale = 0
if len(sys.argv) >= 4:
	flag = int(sys.argv[3])
	if flag in [0,1]:
		rescale = flag

# constants
topn = 100
minwordlen = 3

# read in stop words
stopwords = {}
f = open('stopwords.txt', 'r')
line = f.readline()
while line != '':
	w = line.strip()
	stopwords[w] = 0
	line = f.readline()
f.close()

# read jokes
wordRegex = re.compile(r'\W+')
jokes = {}
words = {}
f = open('joketext.txt', 'r')
line = f.readline()
while line != '':
	row = line.split('\t')
	id = int(row[0])
	text = row[1].strip().lower()
	jokes[id] = {'count':0, 'words':{}}
	for w in wordRegex.split(text):
		if w == '':
			continue
		elif len(w) < minwordlen:
			continue
		elif w in stopwords:
			stopwords[w] += 1
			continue
		if w not in words:
			words[w] = 0
		words[w] += 1
		if w not in jokes[id]['words']:
			jokes[id]['words'][w] = 0
		jokes[id]['words'][w] += 1
		jokes[id]['count'] += 1
	line = f.readline()
f.close()
print 'Size of dictionary: {}'.format(len(words))
srt = sorted(words, key=words.get, reverse=True);
nwords2disp = min(topn, len(srt))
print 'Top {} words: {}'.format(nwords2disp,[(srt[i],words[srt[i]]) for i in range(nwords2disp)])

# compute similarities
minsim = 1
maxsim = 0
avgsim = 0
nnz = 0
sim = {}
for j in jokes:
	norm = 0
	for w in jokes[j]['words']:
		c = jokes[j]['words'][w]
		norm += c * c
	jokes[j]['norm'] = math.sqrt(norm)
for j1 in jokes:
	sim[j1] = {}
	for j2 in jokes:
		if j1 == j2:
			continue
		s = 0
		for w in jokes[j1]['words']:
			if w in jokes[j2]['words']:
				s += jokes[j1]['words'][w] * jokes[j2]['words'][w]
		s /= jokes[j1]['norm'] * jokes[j2]['norm']
		avgsim += s
		if s < minsim: minsim = s
		if s > maxsim: maxsim = s
		if s != 0:
			sim[j1][j2] = s
			nnz += 1
avgsim /= nnz #len(jokes) * (len(jokes)-1)
var = 0
for j1 in sim:
	for j2 in sim[j1]:
		var += (sim[j1][j2] - avgsim) * (sim[j1][j2] - avgsim)
var /= nnz #len(jokes) * (len(jokes)-1)
stdsim = math.sqrt(var)
print 'Minimum similarity: {}'.format(minsim)
print 'Maximum similarity: {}'.format(maxsim)
print 'Average similarity: {}'.format(avgsim)
print 'Std Dev similarity: {}'.format(stdsim)

# threshold (and rescale?)
if threshold > 0:
	thresh = avgsim
	if threshold == 2:
		thresh += stdsim
	range = maxsim - thresh
	newminsim = 1
	newmaxsim = 0
	newavgsim = 0
	nnz = 0
	for j1 in sim:
		for j2 in sim[j1]:
			s = sim[j1][j2]
			if rescale == 1:
				s = (s - thresh) / range
				s = min(1, max(0, s))			
			else:
				if s < thresh:
					s = 0
			sim[j1][j2] = s
			if s < newminsim: newminsim = s
			if s > newmaxsim: newmaxsim = s
			newavgsim += s
			if s != 0: nnz += 1
	newavgsim /= nnz #len(jokes) * (len(jokes)-1)
	var = 0
	for j1 in sim:
		for j2 in sim[j1]:
			var += (sim[j1][j2] - newavgsim) * (sim[j1][j2] - newavgsim)
	var /= nnz #len(jokes) * (len(jokes)-1)
	newstdsim = math.sqrt(var)
	print 'New Minimum similarity: {}'.format(newminsim)
	print 'New Maximum similarity: {}'.format(newmaxsim)
	print 'New Average similarity: {}'.format(newavgsim)
	print 'New Std Dev similarity: {}'.format(newstdsim)

# write to file
f = open(outFile,'w')
for j1 in sim:
	for j2 in sim[j1]:
		if sim[j1][j2] != 0:
			f.write('{}\t{}\t{}\n'.format(j1,j2,sim[j1][j2]))
f.close()
