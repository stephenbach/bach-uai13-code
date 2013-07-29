#!/bin/sh

echo "Compiling..."
mvn compile > /dev/null
mvn dependency:build-classpath -Dmdep.outputFile=classpath.out > /dev/null
mkdir output > /dev/null

echo "Running HL-MRF-Q on Citeseer..."
java -Xmx1g -cp ./target/classes:`cat classpath.out` edu.umd.cs.bachuai13.CollectiveClassification citeseer quad > output/citeseer.hlmrf-q.out
echo "Running HL-MRF-L on Citeseer..."
java -Xmx1g -cp ./target/classes:`cat classpath.out` edu.umd.cs.bachuai13.CollectiveClassification citeseer linear > output/citeseer.hlmrf-l.out
echo "Running MRF on Citeseer..."
java -Xmx1g -cp ./target/classes:`cat classpath.out` edu.umd.cs.bachuai13.CollectiveClassification citeseer bool > output/citeseer.mrf.out

echo "Running HL-MRF-Q on Cora..."
java -Xmx1g -cp ./target/classes:`cat classpath.out` edu.umd.cs.bachuai13.CollectiveClassification cora quad > output/cora.hlmrf-q.out
echo "Running HL-MRF-L on Cora..."
java -Xmx1g -cp ./target/classes:`cat classpath.out` edu.umd.cs.bachuai13.CollectiveClassification cora linear > output/cora.hlmrf-l.out
echo "Running MRF on Cora..."
java -Xmx1g -cp ./target/classes:`cat classpath.out` edu.umd.cs.bachuai13.CollectiveClassification cora bool > output/cora.mrf.out

echo "Processing results..."
cd src/main/matlab
matlab -nodesktop -nosplash -r parse_cc
