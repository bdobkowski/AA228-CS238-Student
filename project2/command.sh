#!/bin/bash
# /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project2/project2.py /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project2/data/small.csv /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project2/output/small.policy | tee /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project2/logs/small.log;	
# echo "finished small problem"
# /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project2/project2.py /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project2/data/medium.csv /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project2/output/medium.policy | tee /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project2/logs/medium.log;	
# echo "finished medium problem"
/home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project2/project2.py /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project2/data/large.csv /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project2/output/large.policy | tee /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project2/logs/large.log;	
echo "finished large problem"
echo "bash script finished running"