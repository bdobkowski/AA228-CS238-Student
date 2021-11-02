#!/bin/bash
/home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project1/project1.py /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project1/data/small.csv /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project1/output/small2.gph | tee /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project1/logs/small2.log;
echo "finished small graph"
/home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project1/project1.py /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project1/data/medium.csv /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project1/output/medium2.gph | tee /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project1/logs/medium2.log;
echo "finshed medium graph"
# /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project1/project1.py /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project1/data/large.csv /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project1/output/large2.gph | tee /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project1/logs/large2.log;
# echo "finished large graph"
echo "bash script finished running"