# LLM Textmarker
This repo contains all relevant code for the creation of a so called *LLM Textmarker* which refers to
the ability of an LLM to detect the start and end of different topics within a text.

![Textmarker UI](./assets/LLM_Textmarker_UI.png)

## Dataset
The /data-folder contains all necessary files for the creation of the CV dataset.
The different sections of the CV are separately created and then combined together under particular constraints to ensure logical consistency of the content.

# Training
The /training folder contains a notebook that prepares and starts the LLM Training within AWS Sagemaker.
The training script which is then run separately on the GPU (started from the notebook) is included in the
/scripts folder within the /training folder. The instance is g5.2xlarge and the base LLM is
mistral-7b-0.2-instruct. NOTE: mistral-7b-0.2-instruct is a gated repo on HuggingFace which means that 
you need to authenticate towards HuggingFace when using it. You need to set the env-variable HF-TOKEN for that.

## Evaluation
The /evaluation-folder contains the files for evaluating the Json-Output of the LLM by remodelling it as a NER-style task.
Additionally there is an example notebook for building up an argilla workspace to visually inspect the model predictions
and potentially start an annotation workflow.

## Deployment (and Inference)
The /code-folder within the /inference-folder contains the specific files that are required 
for deploying a custom Sagemaker endpoint (a custom model and a custom inference workflow).

## Gradio UI
The /app folder contains the above displayed simple Gradio UI that display the core functionality
of marking different topics in an unstructured text.