# SurveyKeywordsExtraction
Question Keywords Extraction 

## Install 
* Requirement:
  * miniconda: https://docs.conda.io/en/latest/miniconda.html
* Install conda environment: conda env create -f environment.yml
* Active conda environment: conda activate surveyKeywords
* Download pre-trained ELMO: python getPerpare.py

## Train model with IRRI annotated data:
* Train IRRI annotated excel file: python irriDataTrain.py pathToIRRIdata.xlsx
* Training with GPU support: python irriDataTrain.py path/ToIRRIdata.xlsx --gpu
* The IRRI annotated excel file should have following fields:
  * Survey term: Survey questions
  * PO_0009010: Ontology classes
  * Term: Ontology terms
  * Options: options for survey questions

## Apply model:
* Apply model to question keywords extraction: python ifpriQuestionKeywordsExtraction.py path2Question.xlsx output.tsv
* The path2Question.xlsx should contain 1 field:
  * Survey term: Survey questions
* output.tsv is the output file in format: Column 1 \t Column 2
  * Column 1: question
  * Column 2: top 3 extracted keywords
