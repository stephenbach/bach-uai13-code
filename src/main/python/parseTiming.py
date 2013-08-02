#!/usr/bin/python

import sys, re

def getTime(method):
	return  str(round(sum(times[method]) / len(times[method]), 2))

# Defines files to read from
files = {}
files['citeseer.hlmrf-q'] = 'output/timing.citeseer.hlmrf-q.out'
files['citeseer.hlmrf-l'] = 'output/timing.citeseer.hlmrf-l.out'
files['citeseer.mrf'] = 'output/timing.citeseer.mrf.out'
files['cora.hlmrf-q'] = 'output/timing.cora.hlmrf-q.out'
files['cora.hlmrf-l'] = 'output/timing.cora.hlmrf-l.out'
files['cora.mrf'] = 'output/timing.cora.mrf.out'
files['epinions.hlmrf-q'] = 'output/timing.epinions.hlmrf-q.out'
files['epinions.hlmrf-l'] = 'output/timing.epinions.hlmrf-l.out'
files['epinions.mrf'] = 'output/timing.epinions.mrf.out'

# Defines lists of times for each method,problem pair
times = {}
for method in files:
	times[method] = []
	
startRegex = re.compile(r'(\d+) \[main\] INFO  edu.umd.cs.psl.application.inference.MPEInference  - Beginning inference.')
endRegex = re.compile(r'(\d+) \[main\] INFO  edu.umd.cs.psl.application.inference.MPEInference  - Inference complete. Writing results to Database.')

for method, timeList in times.iteritems():
	f = open(files[method],'r')
	line = f.readline()
	method = ''
	startTime = 0
	while line != '':
		matches = startRegex.search(line)
		if matches != None:
			# found start time
			if startTime != 0:
				print 'ERROR: Found two consecutive start times without end time in between.'
				exit()
			startTime = long(matches.group(1))
		matches = endRegex.search(line)
		if matches != None:
			# found end time
			if startTime == 0:
				print 'ERROR: Found end time without matching start time.'
				exit()
			endTime = long(matches.group(1))
			elapsed = endTime - startTime
			# add to results
			timeList.append(0.001 * elapsed)
			# Resets state
			startTime = 0
		line = f.readline()
	f.close()

# output
print 'BEGIN TIMING RESULTS TABLE'
print
print '\\begin{tabular}{lrrr}'
print '\\toprule'
print ' & Citeseer & Cora & Epinions \\\\'
print '\midrule'
print 'HL-MRF-Q & ' + getTime('citeseer.hlmrf-q') + ' & ' + getTime('cora.hlmrf-q') + ' & ' + getTime('epinions.hlmrf-q') + ' \\\\'
print 'HL-MRF-L & ' + getTime('citeseer.hlmrf-l') + ' & ' + getTime('cora.hlmrf-l') + ' & ' + getTime('epinions.hlmrf-l') + ' \\\\'
print 'MRF & ' + getTime('citeseer.mrf') + ' & ' + getTime('cora.mrf') + ' & ' + getTime('epinions.mrf') + ' \\\\'
print '\\bottomrule'
print '\\end{tabular}'
print
print 'END TIMING RESULTS TABLE'
