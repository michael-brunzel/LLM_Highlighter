# This script generates samples for a particular section of the CV
# The potential sections for the CV are academic records/ education/ hobbies/ personal infos/ skills/ work experience 
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
import numpy as np
from rouge_score import rouge_scorer
import json
import time
import re
import argparse
from data.name_list import names
from data.origin import origin_names
from prompt_templates.education import edu_prompt1, edu_prompt2
from prompt_templates.personal_infos import personal_prompt1, personal_prompt2
from prompt_templates.work_experience import work_prompt, work_prompt_multi2, work_prompt_sentences
from prompt_templates.skills import skills_prompt1, skills_prompt2
from prompt_templates.academia import academia_prompt1, academia_prompt2
from prompt_templates.hobbies import hobbies_prompt1, hobbies_prompt2


load_dotenv()

POSITIONS = ["Banker", "Lawyer", "Data Scientist", "Data Engineer", "Solution Architect",
             "Business Analyst"]
NATIONS = ["German", "French", "Italian", "British", "American", "Australian", "Spanish", "Finnish", "Swedish"]
SENTENCE_LENGHTS = ["two", "three", "four"]
GRAD_YEARS = ["2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014",
              "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022"]
HOBBIES = ["Mountainbiking", "Golf", "Football", "Dancing", "Skiing", "Swimming", "Climbing"]

CAT2PROMPTS = {
    "work_experience": [work_prompt, work_prompt_multi2, work_prompt_sentences],
    "skills": [skills_prompt1, skills_prompt2],
    "personal": [personal_prompt1, personal_prompt2],
    "education": [edu_prompt1, edu_prompt2],
    "academia": [academia_prompt1, academia_prompt2],
    "hobbies": [hobbies_prompt1, hobbies_prompt2]
}

def run_oai_requests(messages: list, category: str, llm_kwargs: dict) -> None:
    """
    Run the requests with the OpenAI API and collect the response. 

    Args:
      messages (list):
      category (str): The name of the category to which the prompt results belong to.
      llm_kwargs (dict): The parameters for the LLM
    """
    
    client = OpenAI()


    # Add code for accessing the OpenAI API in a programmatic way
    response_list = []
    rouge_scores = []

    for message in messages: #[20:]: #[messages[0]]*2: #[:2]:
        retry_cnt = 0
        backoff_time = 30
        while retry_cnt <= 3:
          try:
            print(message)
            response = client.chat.completions.create(
                messages=message["prompt"],
                **llm_kwargs
                )
            print(response)
            raw_response = response.choices[0].message.content
            response_dict = {
              "prompt": message["prompt"][1]["content"],
              "response": raw_response,
              "category": category,
              "name": message["meta"].get("name", ""),
              "position": message["meta"].get("position", ""),
              "grad_year": message["meta"].get("grad_year", ""),
            }
            txt_responses = [single_dict["response"] for single_dict in response_list]
            rouge_scores = [scorer.score(raw_response, txt) for txt in txt_responses]
            rouge_scores = [score["rougeL"].fmeasure for score in rouge_scores]
            if len(rouge_scores) > 0:
              print(max(rouge_scores))
            # If rouge score between new sample and existing samples is too high,
            # no additional information content
            if len(rouge_scores) > 0 and max(rouge_scores) > 0.6:
                print("Not adding sample due to high similarity")
                break
            response_list += [response_dict]

            with open(f"./prompt_results/{category}.json", "w") as fp:
                json.dump({"prompt_results": response_list}, fp, ensure_ascii=False, indent=4)
            break
          except OpenAIError as e:
              print(f"OpenAIError: {e}.")
              print(f"Retrying in {backoff_time} seconds...")
              time.sleep(backoff_time)
              backoff_time *= 1.5
              retry_cnt += 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("category", type=str, help="Which category should be created?")

    args = parser.parse_args()
    category = args.category

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    llm_params = {
        "model": "gpt-3.5-turbo-0125",
        "max_tokens": 256,
        "top_p": 0.5,
        "temperature": 1.0,
        "frequency_penalty": 0,
        "presence_penalty": 2,
        "n": 1,
        "stop": ["\n16", "16.", "16 ."]       
    }

    # prompt_list = [edu_prompt.format(position="banker", grad_year=2013, nation="Italian")]
    names = re.findall(r"(?:\d+.\s)(.*)", names)
    origins = re.findall(r"(?:\d+.\s)(.*)", origin_names)


    # Sample the writing style, the number of sentence and the example in the prompt (create list of indices for the prompts)
    prompt_list = []
    meta_list = []
    for _ in range(100):
        field2idx = {
            "position": POSITIONS[np.random.choice(len(POSITIONS), 1).tolist()[0]],
            "grad_year": GRAD_YEARS[np.random.choice(len(GRAD_YEARS), 1).tolist()[0]],
            "sentence_length": SENTENCE_LENGHTS[np.random.choice(len(SENTENCE_LENGHTS), 1).tolist()[0]],
            "name": names[np.random.choice(len(names), 1).tolist()[0]],
            "nation": NATIONS[np.random.choice(len(NATIONS), 1).tolist()[0]],
            "prompt": CAT2PROMPTS[category][np.random.choice(len(CAT2PROMPTS[category]), 1).tolist()[0]],
            "origin": origins[np.random.choice(len(origins), 1).tolist()[0]],
            "hobby": HOBBIES[np.random.choice(len(HOBBIES), 1).tolist()[0]]
        }
        meta_dict = {}
        existing_placeholders = re.findall(r"{(\w+)}", field2idx["prompt"])
        for field in ["position", "name", "grad_year", "sentence_length", "nation", "origin", "hobby"]:
            if field in existing_placeholders:
                meta_dict[field] = field2idx[field] # add the sampled value here
        final_prompt = field2idx["prompt"].format(**meta_dict) 
        prompt_list += [final_prompt]
        meta_list += [meta_dict]

    # Store prompts and the corresponding meta data
    messages = [{
        "prompt":
            [{
                "role": "assistant",
                "content": "You are an AI assistant that helps to generate a dataset"
            },
            {
                "role": "user",
                "content": prompt
            }],
        "meta": meta
    }
    for prompt, meta in zip(prompt_list, meta_list)]

    run_oai_requests(messages, category, llm_params)