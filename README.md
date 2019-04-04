# table_extractor
Code and data used in the paper, A Machine Learning Approach to Zeolite Synthesis Enabled by Automatic Literature Data Extraction

There are two main components to this repository:
1. table_extractor code
2. zeolite synthesis data

# 1. Table Extraction Code
This code extracts tables into json format from HTML/XML files. These HTML/XML files need to be supplied by the researcher. The code is written in Python3. To run the code:
1. Fork this repository
2. Download the Olivetti group materials science FastText word embeddings
    - Available here: https://figshare.com/s/70455cfcd0084a504745
    - Download all 4 files and place in the tableextractor/bin folder
3. Install all dependencies
    - json, pandas, spacy, bs4, gensim, numpy, unidecode, sklearn, scipy, traceback
4. Use Jupyter (Table Extractor Tutorial) to run the code

The code takes in a list of files and corresponding DOIs and returns a list of all tables extracted from the files as JSON objects. Currently, the code supports files from ACS, APS, Elsevier, Wiley, Springer, and RSC. 

# 2. Zeolite Synthesis Data
The germanium containing zeolite data set used in the paper is publicly available in both Excel and CSV formats. Here is a description of each feature:

doi- DOI of the paper the synthesis route comes from

Si:B- molar amount of each element/compound/molecule used in the synthesis. Amounts are normalized to give Si=1 or Ge=1 if Si=0

Time- crystallization time in hours

Temp- crystallization temperature in °C

SDA Type- name given to the organic structure directing agent (OSDA) molecule in the paper

SMILES- the SMILES representation of the OSDA molecule

SDA_Vol- the DFT calculated molar volume of the OSDA molecule in bohr^3

SDA_SA- the DFT calculated surface area of the OSDA molecule in bohr^2

SDA_KFI- the DFT calculated Kier flexibility index of the OSDA molecule

From?- the location within a paper the compositional information is extracted. Either Table, Text, or Supplemental

Extracted- Products of the synthesis as they appear in the paper

Zeo1- the primary zeolite (zeotype) material made in the synthesis

Zeo2- the secondary zeolite (zeotype) material made in the synthesis

Dense1- the primary dense phase made in the synthesis

Dense2- the secondary dense phase made in the synthesis

Am- whether an amorphous phase is made in (or remains after) the synthesis

Other- any other unidentified phases made in the synthesis

ITQ- whether the synthesis made a zeolite in the ITQ series

FD1- the framework density of Zeo1

MR1- the maximum ring size of Zeo1

FD2- the framework density of Zeo2

MR2- the framework density of Zeo2

# Citing
If you use this code or data, please cite the following as appropriate. 

[Citiation coming soon]
