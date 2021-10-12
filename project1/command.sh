#!/bin/bash
/home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project1/project1.py /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project1/data/small.csv /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project1/output/small.gph >& /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project1/logs/small.log;
echo "finished small graph"
/home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project1/project1.py /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project1/data/medium.csv /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project1/output/medium.gph >& /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project1/logs/medium.log;
echo "finshed medium graph"
/home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project1/project1.py /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project1/data/large.csv /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project1/output/large.gph >& /home/bdobkowski/Stanford/AA228/AA228-CS238-Student/project1/logs/large.log;
echo "finished large graph"
echo "bash script finished running"