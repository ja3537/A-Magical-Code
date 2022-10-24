# Project 3: A Magical Code

## Citation and License
This project belongs to Department of Computer Science, Columbia University. It may be used for educational purposes under Creative Commons **with proper attribution and citation** for the author TAs **Joe Adams (first author), and the Instructor, Prof. Kenneth Ross**.

## Summary

Course: COMS 4444 Programming and Problem Solving (Fall 2022)  
Problem Description: http://www.cs.columbia.edu/~kar/4444f22/node20.html    
Course Website: http://www.cs.columbia.edu/~kar/4444f22/4444f22.html  
University: Columbia University  
Instructor: Prof. Kenneth Ross  
Project Language: Python


### TA Designer for this project

Joe Adams

### Teaching Assistants for Course
1. Joe Adams
1. Rohit Gopalakrishnan

## Installation

Requires **python3.9** or higher.

Also requires numpy.


## Usage

### Simulator

```bash
python main.py
```

## Optional Flags

```bash
usage: main.py [-h] [--seed SEED] [--agent AGENT] [--messages MESSAGES] [--output OUTPUT] [-n N] [--null_rate NULL_RATE] [--verbose] [--rand_n]

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED, -s SEED  seed used by random number generator. Default is 42. Specify 0 to generate random seed.
  --agent AGENT, -a AGENT
                        which agent to use from 1 to 8. d for default
  --messages MESSAGES, -m MESSAGES
                        name of file holding messages to encode
  --output OUTPUT, -o OUTPUT
                        name of the output file where decoded messages and scores are stored
  -n N                  set number of shuffles per deck for this mission
  --null_rate NULL_RATE, -nr NULL_RATE
                        set the probablility (between 0 and 1) of giving agent a random deck instead of a message
  --verbose, -v         verbose mode includes encrypted and shuffled decks in output
  --rand_n, -rn         (boolean) sets n to be a random value up to N for each deck


  
```

## Other Information

Decks are represented as a list of integers from 0 to 51. 

Agent encode function is given a string and should return a legal deck.

Agent decode function is given a deck and should return a string. 
This string should be "NULL" if the agent believes this is a random deck.    
This string should begin with "PARTIAL: " if the agent believes they are returning a partial message (single space after the colon is removed from message).
