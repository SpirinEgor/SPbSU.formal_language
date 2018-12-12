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

# Install required packages
```
pip install matplotlib
```

# Data
**All indexes starts with 1**

Contex-Free grammar defines in chomsky normal form:
```
head tail1
head eps
head tail1 tail2
...
```
Graph defines by its edges:
```
from symbol to
... 
```
Instead of printing all matrix for output, print only non-empty cells for each nonterm symbol:
```
nonterm1 row1 col1 row2 col2 ...
nonterm2
...
```
## Example
Chomsky Normal Form:
```
E num
S2 +
S2 *
L (
R )
E E S1
E L N
S1 S2 E
N E R
```
Graph:
```
1 num 2,
2 + 3,
3 num 4,
2 * 1
```
Result:
```
L
R
S2 1 2 2 1 2 3 3 4
S1
E
N
```
