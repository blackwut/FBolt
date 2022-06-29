#!/bin/bash
cd FVecSum
storm kill -w 0 FVecSum
mvn clean install
storm jar ./target/FVecSum-1.0.jar FVecSum.FVecSumTopology
cd ..
