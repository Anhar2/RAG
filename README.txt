#Implementing RAG Pipeline for Career Guidance


A full pipeline of the Retrieval Augmented Generation system designed for the task of generating skills recommendations. Given a dataset, the system makes vector stores, then given the user query which consists of the job title they are seeking guidance to haunt. 






## Repo Files
* Job_skills.py; the main clean well-structured script, you can run directly as mentioned below.
* Job_skills.ipynb; the notebook is doing the same job as the python script, added for smoother scrutinizing, also it contains some of other either successful or failed attempts in addition to some among the best outputs and their scores “please read the report for that”.
* requirements.txt
* output.txt; a sample output containing the generated text and its scores.






### How to run it:
* pip install -r requirements.txt
* export HF_TOKEN “your huggingface token” 
* Python3 job_skills.py -q ‘user query’ -k ‘your k, default to 4’ -d ‘dataset path and name, default to sampled_data.csv assuming it’s in the same dir’