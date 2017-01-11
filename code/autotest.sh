#!/bin/bash
javac NLP.java
java NLP
perl score train.txt test.answer.txt out.txt > result.txt