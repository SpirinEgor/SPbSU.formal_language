# SPBU.formal_language
Intersection of regular language as graph and CFG in Chomsky normal form

# Guide for using CI
CI testing is composed of 2 parts:
1. Build dependecies  
The docker image run `./install.sh` where you should define how to install all your dependencies.  
There are commands: `curl`, `wget`, `git`, `apt`...

2. Test your code  
There is `runner.py`, which prepares tests and run your code using `run.sh`, where you should define, how to run your script.  
Use 3 variables to get paths to input and output files.

## Data
Contex-Free grammar defines in chomsky normal form:
```
head : tail1 tail2
head : tail1
...
```
Graph defines by its edges:
```
from to symbol
... 
```
Instead of printing all matrix for output, print only non-empty cells with all nonterms in it:
```
row col nonterm1 nonterm2 ...
...
```
## Example
Chomsky Normal Form:
```
E : E S1
E : num
E : L N
S1 : S2 E
S2 : +
S2 : *
L : (
N : E R
R : )
```
Graph:
```
0 1 num
1 2 +
2 3 num
1 0 *
```
Result:
```
0 1 E
0 3 E
1 0 S2
1 1 S1
1 2 S2
1 3 S1
2 3 E
```
