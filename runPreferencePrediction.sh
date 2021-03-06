#!/bin/sh

echo "Compiling..."
mvn compile > /dev/null
mvn dependency:build-classpath -Dmdep.outputFile=classpath.out > /dev/null
mkdir output > /dev/null

echo "Running HL-MRF-Q on Jester..."
java -Xmx12g -cp ./target/classes:`cat classpath.out` edu.umd.cs.bachuai13.jester.Jester quad > output/jester.hlmrf-q.out
echo "Running HL-MRF-L on Jester..."
java -Xmx12g -cp ./target/classes:`cat classpath.out` edu.umd.cs.bachuai13.jester.Jester linear > output/jester.hlmrf-l.out
echo "Running BPMF on Jester..."
cd src/main/matlab
matlab -nodesktop -nosplash -r run_bpmf_jester > /dev/null

echo "Processing results..."
matlab -nodesktop -nosplash -r parse_jester
