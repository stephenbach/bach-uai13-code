#!/usr/bin/python

import random as rand
import sys

# parse input
njokes = 100
nusers = 1000
if len(sys.argv) > 1:
	nusers = int(sys.argv[1])
nround = 1
if len(sys.argv) > 2:
	nround = int(sys.argv[2])
pctobs = 0.25
if len(sys.argv) > 3:
	pctobs = float(sys.argv[3])
nobs = int(pctobs * njokes * nusers)

# read data
R = []
f = open('jester-1.csv','r')
line = f.readline()
while line != '':
	row = line.strip().split(',')
	nr = int(row[0])
	if nr == njokes:
		R_user = [(float(r)+10)/20 for r in row[1:(njokes+1)]]
# 		if len(R_user) != njokes:
# 			print 'Parsing error: incorrect number of ratings'
# 			exit()
		R.append(R_user)
	line = f.readline()
f.close()
# print 'Number of users: {0}'.format(len(R))

# seed random for deterministic splits
rand.seed(311)

# randomly sample users
rand.shuffle(R)
R = [R[0:nusers],R[nusers:(2*nusers)]]

# create train/test observed/unobserved folds
label = ['tr','te']
for s in range(2):
	# pivot matrix to list
	Rlist = [(s*nusers+u+1,i+1,R[s][u][i]) for u in range(nusers) for i in range(njokes)]
	# print min([r[2] for r in Rlist])
	# print max([r[2] for r in Rlist])

	# sample uniform random observed/unobserved set
	for round in range(nround):
		rand.shuffle(Rlist)
		obs = Rlist[0:nobs]
		uno = Rlist[nobs:len(Rlist)]
		# write to files
		f = open('jester-1-{0}-obs-{1}.txt'.format(label[s],round),'w')
		for r in obs:
			f.write('{0}\t{1}\t{2}\n'.format(r[0],r[1],r[2]))
		f.close()
		f = open('jester-1-{0}-uno-{1}.txt'.format(label[s],round),'w')
		for r in uno:
			f.write('{0}\t{1}\t{2}\n'.format(r[0],r[1],r[2]))
		f.close()

