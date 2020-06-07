#!/bin/bash

for afile in external/*-csg-tree-prediction.svg
do
    python utils/pdflatex2pdf.py ${afile}
done
