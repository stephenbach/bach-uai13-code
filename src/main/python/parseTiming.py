#!/usr/bin/python

import sys, re

if len(sys.argv) < 2:
	print 'USAGE: python parseTiming.py <logFile1 logFile2 ...>'
	exit()
	
methodRegex = re.compile(r'Learned model (.+)\n')
startRegex = re.compile(r'(\d+) \[main\] INFO  edu.umd.cs.psl.application.inference.MPEInference  - Beginning inference.')
endRegex = re.compile(r'(\d+) \[main\] INFO  edu.umd.cs.psl.application.inference.MPEInference  - Inference complete. Writing results to Database.')

results = {}
for i in range(1,len(sys.argv)):
	logFile = sys.argv[i]
	f = open(logFile,'r')
	line = f.readline()
	method = ''
	startTime = 0
	while line != '':
		matches = methodRegex.search(line)
		if matches != None:
			if method is not '':
				print 'ERROR: Found two consecutive method names without inference times'
				exit()
			method = matches.group(1)
		matches = startRegex.search(line)
		if matches != None:
			# found start time
			if startTime != 0:
				print 'ERROR: Found two consecutive start times without end time in between.'
				exit()
			# parse time
			#tokens = line.split(' ')
			startTime = long(matches.group(1))
		matches = endRegex.search(line)
		if matches != None:
			# found end time
			if startTime == 0:
				print 'ERROR: Found end time without matching start time.'
				exit()
			# parse time
			#tokens = line.split(' ')
			endTime = long(matches.group(1))
			elapsed = endTime - startTime
			# add to results
			if method not in results:
				results[method] = []
			results[method].append(0.001 * elapsed)
			# Resets state
			method = ''
			startTime = 0
		line = f.readline()
	f.close()

# output
for method,times in results.iteritems():
	# compute average
	avgTime = sum(times) / len(times)
	print '{} \t{}'.format(method,avgTime)
