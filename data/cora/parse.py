#!/usr/bin/python

import sys

fpfx = 'cora'
idmap = {}
haslabel = {}
hasword = {}
labels = {'Case_Based':1
		 ,'Genetic_Algorithms':2
		 ,'Neural_Networks':3
		 ,'Probabilistic_Methods':4
		 ,'Reinforcement_Learning':5
		 ,'Rule_Learning':6
		 ,'Theory':7
		 }

f = open('{}.content'.format(fpfx),'r')
line = f.readline()
id = 0
while line != '':
	row = line.strip().split('\t')
	id += 1
	idmap[row[0]] = id
	lab = row[len(row)-1]
	haslabel[id] = labels[lab]
	hasword[id] = {}
	for i in range(1,len(row)-1):
		if row[i] == '1':
			hasword[id][i] = 1
	line = f.readline()
f.close()

f = open('{}.labels'.format(fpfx),'w')
for id in haslabel:
	f.write('{}\t{}\n'.format(id,haslabel[id]))
f.close()

f = open('{}.words'.format(fpfx),'w')
for id in hasword:
	f.write('{}\t'.format(id))
	for w in hasword[id]:
		f.write('{}:1 '.format(w))
	f.write('\n')
f.close()

fi = open('{}.cites'.format(fpfx),'r')
fo = open('{}.links'.format(fpfx),'w')
line = fi.readline()
while line != '':
	row = line.strip().split('\t')
	if row[0] in idmap and row[1] in idmap:
		id1 = idmap[row[0]]
		id2 = idmap[row[1]]
		if id1 != id2:
			fo.write('{}\t{}\n'.format(id1,id2))
	line = fi.readline()
fi.close()
fo.close()

f = open('{}.ids'.format(fpfx),'w')
for origid in sorted(idmap):
	f.write('{}\t{}\n'.format(origid,idmap[origid]))
f.close()
