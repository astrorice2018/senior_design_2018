#!/bin/bash

for i in 1 2 3 4
do
    python3 detect1.py --images $1'patch'$i --det $1'patch'$i'/image_det'
done
