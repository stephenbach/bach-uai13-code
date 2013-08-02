#!/bin/sh

echo "Compiling..."
mvn compile > /dev/null
mvn dependency:build-classpath -Dmdep.outputFile=classpath.out > /dev/null
mkdir output > /dev/null

echo "Running HL-MRF-Q on Caltech left..."
java -Xmx16g -cp ./target/classes:`cat classpath.out` edu.umd.cs.bachuai13.vision.ImageReconstruction caltech left random > output/caltech.left.hlmrf-q.out
echo "Running HL-MRF-Q on Caltech bottom..."
java -Xmx16g -cp ./target/classes:`cat classpath.out` edu.umd.cs.bachuai13.vision.ImageReconstruction caltech bottom random > output/caltech.bottom.hlmrf-q.out
echo "Running HL-MRF-Q on Olivetti left..."
java -Xmx16g -cp ./target/classes:`cat classpath.out` edu.umd.cs.bachuai13.vision.ImageReconstruction olivetti left random > output/olivetti.left.hlmrf-q.out
echo "Running HL-MRF-Q on Olivetti bottom..."
java -Xmx16g -cp ./target/classes:`cat classpath.out` edu.umd.cs.bachuai13.vision.ImageReconstruction olivetti bottom random > output/olivetti.bottom.hlmrf-q.out

echo "Processing results..."
