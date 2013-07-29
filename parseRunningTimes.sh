#!/bin/sh

echo "Parsing running times for collective classification"
echo "and social-trust prediction..."
src/main/python/parseTiming.py output/citeseer.hlmrf-l.out output/citeseer.mrf.out
