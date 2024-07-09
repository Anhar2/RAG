#!/usr/bin/env python

 

import os
import nltk
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from huggingface_hub import notebook_login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score
import argparse


class RAG_Pipeline():
    def __init__(self, args):
        self.data_path = args.documents_path #Full including the dataset name
        self.job_title = args.query
        self.k = int(args.k)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        jobs = self.data_preparation()
        index = self.retrieval_component( jobs)
        context = self.vector_search(self.job_title, jobs, index)
        output_text_rel = self.generation_componenet( context)
        metrics = self.evaluation( output_text_rel, context)
        with open('output.txt', 'w') as f:
            f.write(output_text_rel + "\n")
        with open('output.txt', 'a') as f:
            f.write(str(metrics))


    def data_preparation(self):
        print("Preparing data...")
        def preprocess_text( text):
            # Convert to lowercase
            text = text.lower()
            # Remove punctuation and html tags
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = re.sub('<[^>]+>', '', text)


            return  text
        #Assuming the dataset would always be with the same titles
        jobs = pd.read_csv(self.data_path, usecols=['job_title', 'description', 'requirements', 'career_level'])

        #I have tried it with differnt combinations, and ended up with these three
        jobs['combined'] =  jobs['job_title'].astype(str) + ' ' + jobs['description'].astype(str) + ' ' + jobs['requirements'].astype(str) 

        jobs.fillna('', inplace=True)
        jobs['combined'] = jobs['combined'].apply(preprocess_text)
        

        jobs = jobs[jobs['combined'].apply(lambda x: len(x) > 300)] # Remove short/empty job descriptions and requirements


        #Embedding
        embedding_model = self.embedding_model

        # Encode the concatenated job details into dense vectors
        jobs["embedding"] = jobs['combined'].apply(embedding_model.encode)#tolist())


        return jobs



    def retrieval_component(self, jobs):
        print("Indexing...")
        dimension = self.embedding_model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatL2(dimension)  
        index.add(np.vstack(jobs["embedding"]))  
        return index

    def vector_search(self, user_query, jobs, index):
                """Gets user input query and return top k similar items"""   
                print("Query searching....")             

                def find_similar_jobs(query_embedding, k):
                	"""
                	Finds k most similar jobs based on the job title embedding.
                	
                	:query_embedding: A numpy array representing the embedding of a job title.
                	:k: Number of nearest neighbors to find.
                	:return: Indices of the k nearest job title embeddings.
                	"""
                	D, I = index.search(query_embedding, k)  # D: distances, I: indices
                	return I

                # Generate embedding for the user query
                k = self.k
                query_embedding = self.embedding_model.encode(user_query)
                query_embedding = query_embedding.reshape(1, -1)  # Reshape to match the shape of document_embeddings


                if query_embedding is None:
                    return "Invalid query or embedding generation failed."

                similar_job_indices = find_similar_jobs(query_embedding, k)
                results = jobs["combined"].iloc[list(similar_job_indices[0])].to_dict()
                return results






    def generation_componenet(self, context):
        print("Generation....")
        job_title = self.job_title
        k = self.k

        



        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
        gen_model = AutoModelForCausalLM.from_pretrained(
           "google/gemma-2b-it",
           device_map="auto",
           torch_dtype=torch.bfloat16,
           cache_dir = "/home/carla/.cache/huggingface/hub",
           low_cpu_mem_usage=True

        ).to("cuda")
         



        try:
            del input_ids
            del response
        except NameError:
            pass

        gc.collect()
        torch.cuda.empty_cache()


        # get similar jobs
        #context 

        #The best prompts so far are three
        # 1- Give tailored skills recommendations and career advice to the job seeker for job_title based only on the context provided:
        # 2- Provide skills recommendations and career advice  seeker to job_title based on your knowledge and use the following context for help:
        #3-  Offer personalized skill enhancement suggestions and vocational guidance to the individual seeking employment, focusing exclusively on the job_title, utilizing solely the context at hand.
        


        prompt_template = f"""
        <|system|>
         Give tailored skills recommendations and career advice to the job seeker for job_title based only on the context provided:
         Example Input/Output:
         The LLM should be capable of providing the job seeker with personalized career advice based
         on their targeted job title.
         Targeted Job Title: Machine Learning Engineer
         Personalized Career Advice: Based on your interest in the role of Machine Learning Engineer,
         here are some personalized career advice:
         1. Strengthen your foundation in mathematics and statistics, as they form the backbone of
         machine learning algorithms. Focus on concepts like linear algebra, calculus, probability, and
         optimization techniques.
         2. Enhance your programming skills in languages commonly used in machine learning, such as
         Python and R. Familiarize yourself with libraries like TensorFlow, PyTorch, and scikit-learn for
         implementing machine learning models.
         3. Build a strong portfolio of projects showcasing your expertise in machine learning. Work on
         real-world datasets, develop and deploy machine learning models, and document your process
         and results on platforms like GitHub or Kaggle.
         4. Stay updated with the latest trends and advancements in the field of machine learning. Follow
         research publications, attend conferences, and participate in online courses or workshops to
         expand your knowledge and skills.
         5. Network with professionals in the machine learning community. Join relevant online forums,
         participate in meetups or conferences, and connect with mentors who can provide guidance and
         support in your career journey.
         Remember that continuous learning and practical experience are key to advancing your career
         as a Machine Learning Engineer. Keep exploring new technologies, solving challenging
         problems, and seeking opportunities for growth and development.
        {context}

        <|user|>
        {job_title}
        """





        # recommend is a key word in the current prompt to separate the gen part from the others
        key_word = "Recommendations"

        input_ids = tokenizer(prompt_template, return_tensors="pt").to("cuda")
        response = gen_model.generate(**input_ids, max_new_tokens=500)
        output_text = tokenizer.decode(response[0])
        output_text_rel = output_text[output_text.index(key_word) + len(key_word):]
        return output_text_rel



    def evaluation( self, response, context):
        context = list(context.values())
        context = [[doc for doc in context_document.split()] for context_document in context]
        context = ' '.join(context[0])


        def evaluate_bleu_rouge( candidates, references):
                bleu_score = corpus_bleu(candidates, [references]).score
                scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                rouge_scores = [scorer.score(ref, cand) for ref, cand in zip(references, candidates)]
                rouge1 = sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores)
                return bleu_score, rouge1

        def evaluate_bert_score( candidates, references):
                P, R, F1 = score(candidates, references, lang="en", model_type='bert-base-multilingual-cased')
                return P.mean().item(), R.mean().item(), F1.mean().item()


        candidates = [response]
        references = [context]
        bleu, rouge1 = evaluate_bleu_rouge(candidates, references)
        bert_p, bert_r, bert_f1 = evaluate_bert_score(candidates, references)
        return {"BLEU": bleu,
            "ROUGE-1": rouge1,
            "BERT P": bert_p,
            "BERT R": bert_r,
            "BERT F1": bert_f1,
        }


        






def main():
    argparser = argparse.ArgumentParser(
        description='Retrieval Agumented Generation pipleine')
    argparser.add_argument(
        '-d', '--documents-path',
        type=str,
        default='sampled_jobs.csv', #Assuming it's in the same folder
        help='pass the path of your documents')
    argparser.add_argument(
        '-q', '--query',
        required=True,
        type = str,
        help='Enter the job title the job seeker pursuing')
    
    argparser.add_argument(
        '-k',
        default= 10,
        help='Enter k for KNN search')
    

    args = argparser.parse_args()
    
    key = os.environ['HT_TOKEN']
    import huggingface_hub

    huggingface_hub.login(key)
    RAG_Pipeline(args)


if __name__ == '__main__':

    main()





'''

metrics = evaluate_all( output_text_rel, context) #needs improvements, fine-tuning and increasing the k would be just perfect
#This is where the tuned part of the prompt was " Give tailored skills recommendations and career advice to the job seeker for\
#  job_title based only on the context provided:" without using an example
metrics


# In[79]:


#Same like the previous one, but with k = 1, not logical value to use, but had to try all what my machine can handle
output_text_rel = """
* Software Engineering
* C# Programming
* Distributed Systems
* Cloud Computing
* SQL Database
* Relational Database

**Career Advice:**

* Apply for software engineering positions at companies that align with your skills and interests.
* Network with professionals in the software engineering field.
* Stay updated on the latest trends and technologies in the software engineering field"""
k = 1
context = list(vector_search(job_title, 1).values())
context = ' '.join(context[0]) 
metrics = evaluate_all( output_text_rel, context) 
metrics #Much worse than the above, as expected


# In[41]:


print(output_text_rel)


# In[81]:


#1st without using example with k = 5 as far as I remember 
#
output_text_rel =
**Recommendations:**

**Technical Skills:**

* C# programming language proficiency is essential for this role.
* Familiarity with SQL databases is also highly recommended.
* Experience with cloud platforms such as AWS, Azure, or GCP is a plus.

**Career Advice:**

* Consider pursuing a career in software development or related field.
* Build a strong portfolio of projects that showcase your skills and experience.
* Network with professionals in the software development industry.
* Stay updated on the latest trends and technologies in the software development field.

context = list(vector_search(job_title, 5).values())
context = ' '.join(context[0])
metrics = evaluate_all( output_text_rel, context)
metrics


# In[80]:


#output_text_rel when the 1st prompt contained the example with k=5 propabely 
output_text_rel = """The tailored skills recommendations and career advice for the job seeker are as follows:

**1. Focus on Programming Skills:**
- Enhance your programming skills in languages commonly used in machine learning, such as Python and R.
- Build a strong portfolio of projects showcasing your expertise in machine learning.

**2. Stay Updated with the Latest Trends:**
- Stay updated with the latest trends and advancements in the field of machine learning.
- Follow research publications, attend conferences, and participate in online courses or workshops to expand your knowledge and skills.

**3. Build a Strong Portfolio:**
- Work on real-world datasets, develop and deploy machine learning models, and document your process and results on platforms like GitHub or Kaggle.

**4. Network with Professionals:**
- Join relevant online forums, participate in meetups or conferences, and connect with mentors who can provide guidance and support in your career journey.

**5. Continuous Learning and Growth:**
- Continue exploring new technologies, solving challenging problems, and seeking opportunities for growth and development.<eos>"""
context = list(vector_search(job_title, 5).values())
context = ' '.join(context[0])
metrics = evaluate_all( output_text_rel, context)
metrics


# In[85]:


context


# In[86]:


#1st with k = 10

output_text_rel = """
Based on the context, the recommended skills and career advice for a software engineer would be:

**Technical Skills:**

* Programming languages: Java, Python, C++, SQL
* Frameworks and libraries: Spring Boot, React, Angular
* Cloud platforms: AWS, Azure, Google Cloud Platform
* Data structures and algorithms
* Version control systems (Git)

**Soft Skills:**

* Communication: Excellent written and verbal communication skills
* Teamwork: Ability to work effectively in a team environment
* Problem-solving: Strong analytical and problem-solving skills
* Critical thinking: Ability to think independently and identify solutions
* Time management: Ability to manage multiple tasks and deadlines effectively
* Leadership: Ability to lead and motivate a team

**Career Advice:**

* Start by building a strong foundation in programming languages and data structures.
* Get involved in open-source projects to gain experience and build a network.
* Apply for software engineering internships and entry-level positions.
* Network with professionals in the software engineering field.
* Attend industry conferences and workshops to stay updated on the latest trends.<eos>
"""
#Fortunately, I have the dict of the jobs indices
l = {1628: 'software engineer', 9125: 'software engineer', 13181: 'software engineer', 28871: 'software engineer', 33046: 'software engineer', 35346: 'software engineer', 36349: 'software engineer', 39380: 'software engineer', 13417: 'software development engineer', 27972: 'software development engineer'}
context = []
for i in l.keys():
    context.append(jobs["combined"].iloc[i])
context = ' '.join(context[0])
metrics = evaluate_all( output_text_rel, context)
metrics


# In[76]:


with open('output.txt', 'w') as f:
    f.write(output_text_rel)


# The following are among the best output that I unfortunately couldn't produce due to sudden resources limitations as explained in the report.

# In[90]:


"""
```

Based on the context, the recommended skills and career advice for a software engineer would be:

**Technical Skills:**

* Programming languages: Java, Python, C++, SQL
* Frameworks and libraries: Spring Boot, React, Angular
* Cloud platforms: AWS, Azure, Google Cloud Platform
* Data structures and algorithms
* Version control systems (Git)

**Soft Skills:**

* Communication: Excellent written and verbal communication skills
* Teamwork: Ability to work effectively in a team environment
* Problem-solving: Strong analytical and problem-solving skills
* Critical thinking: Ability to think independently and identify solutions
* Time management: Ability to manage multiple tasks and deadlines effectively
* Leadership: Ability to lead and motivate a team

**Career Advice:**

* Start by building a strong foundation in programming languages and data structures.
* Get involved in open-source projects to gain experience and build a network.
* Apply for software engineering internships and entry-level positions.
* Network with professionals in the software engineering field.
* Attend industry conferences and workshops to stay updated on the latest trends.<eos>
"""


# In[89]:


"""

**Software Engineer**

* **Recommended Skills:** Python, Java, SQL
* **Career Advice:**
    * Focus on building a strong foundation in Python and Java.
    * Develop proficiency in SQL for data manipulation and querying.
    * Consider pursuing certifications in Python and Java to enhance your marketability.
    * Explore opportunities in full-stack development or data science.

**Additional Recommendations:**

* **Learn in-demand technologies:** Stay updated with the latest trends in the software engineering field, such as cloud computing, artificial intelligence, and machine learning.
* **Build a strong network:** Attend industry events, meetups, and conferences to connect with other software engineers and potential employers.
* **Develop problem-solving skills:** Practice solving coding challenges and participate in hackathons to hone your problem-solving abilities.
* **Stay motivated and persistent:** The software engineering field can be challenging, but it's important to stay motivated and persistent in your pursuit of a career in this field.<eos>
"""


# In[ ]:


#  Give tailored skills recommendations and career advice to the job seeker for job_title based only on the context provided:

response = 
**Tailored Skills Recommendations:**

- Object-oriented programming skills
- SQL database skills
- Cloud environment skills
- Agile development methodologies
- Software testing skills
- Design skills

**Career Advice:**

- Seek opportunities in software development companies or startups.
- Focus on building a strong portfolio of high-quality software applications.
- Stay up-to-date with the latest technologies and trends in the software development field.
- Network with other software developers and professionals in the industry.
- Consider pursuing further education or certifications to enhance your skills.<eos>





"""

**Recommended Skills:**

* **Python** is a highly versatile programming language that is widely used in various industries.
* **Java** is another popular programming language that is commonly used for enterprise applications.
* **SQL** (Structured Query Language) is a database programming language that is essential for data manipulation and analysis.

**Career Advice:**

* **Consider pursuing a career in software development or data analytics.** These fields offer excellent job prospects and high earning potential.
* **Look for opportunities to learn and grow in a fast-paced and dynamic environment.**
* **Stay updated on the latest trends and technologies in the software development industry.**

**Additional Recommendations:**

* **Build a strong foundation in computer science concepts.** This can be done through online courses, bootcamps, or self-study.
* **Develop problem-solving and critical thinking skills.** These skills are essential for success in any programming role.
* **Stay connected with the tech community.** Attend meetups, conferences, and online forums to network with other developers and stay informed about industry trends.<eos>
"""

'''
