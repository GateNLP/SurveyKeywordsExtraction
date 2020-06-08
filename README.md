# SurveyKeywordsExtraction
Question Keywords Extraction 

## Install 
* Requirement:
  * miniconda: https://docs.conda.io/en/latest/miniconda.html
* Install conda environment: conda env create -f environment.yml
* Active conda environment: conda activate surveyKeywords
* Download pre-trained ELMO: python getPerpare.py

## Train model:
* Two models are supported for keywords extraction
* 1) irri: irri model is a term matching model that learn attention weights to generate sentence embedding similar to ontology terms
* 2) qsclass: qsclass is a classification model that learn attention weights to generate feature for classification task
* For both models the (words) attention weights are applied for keywords extraction
### Train irri model with IRRI annotation:
* Train IRRI annotated excel file: python modelTrain.py pathToIRRIdata.xlsx
* Training with GPU support: python modelTrain.py path/ToIRRIdata.xlsx --gpu
* The IRRI annotated excel file should have following fields:
  * Survey term: Survey questions
  * PO_0009010: Ontology classes
  * Term: Ontology terms
  * Options: options for survey questions

### Train qsclass model with classifier annotation:
* Train IRRI annotated excel file: python modelTrain.py classifierAnnotated.txt --model qsclass
* Training with GPU support: python modelTrain.py classifierAnnotated.txt --model qsclass --gpu
* The classifier Annotated text file should have following format:
  * 'MajorClass:MinorClass Question'
  * e.g. ENTY:cremat What films featured the character Popeye Doyle ?

## Apply model:
* Apply model to question keywords extraction: python ifpriQuestionKeywordsExtraction.py path2Question.xlsx output.tsv
* The path2Question.xlsx should contain 1 field:
  * Survey term: Survey questions
* output.tsv is the output file in format: Column 1 \t Column 2
  * Column 1: question
  * Column 2: top 3 extracted keywords


## TF.IDF keywords extraction:
* We also experimented the triditional TF.IDF term weights approach:
* Only works for IFPRI formated excel survy 
* Apply TF.IDF: python IFPRIKWE/ultis/tfidf_keywords_extraction_group.py RHoMIS_Minimal_v1.6.xlsx out.tsv
* RHoMIS_Minimal_v1.6.xlsx: is the example excel file
* out.tsv: is the keywords extraction output file
