"""
Sample a coherent CV with the existing snippets for Personal Infos, Work Experience, Skills and Education
and other irrelevant categories. Graduation year and the profession need to be sampled for all categories
(don't include category if nothing is available for this year and profession) if still ambiguous
(same year and profession but different prompt style, pick one randomly..)
"""

import json
import numpy as np
from rouge_score import rouge_scorer
from random import shuffle
from data.build_category_samples import POSITIONS, GRAD_YEARS

SEPARATORS = ["\n", "\n\n"]

if __name__ == "__main__":
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    CV_list = []
    for idx in range(1000):
        print(idx)
        cat2json = {}
        CV_dict = {}
        position = POSITIONS[np.random.choice(len(POSITIONS), 1).tolist()[0]]
        grad_year = GRAD_YEARS[np.random.choice(len(GRAD_YEARS), 1).tolist()[0]]
        separator = SEPARATORS[np.random.choice(len(SEPARATORS), 1).tolist()[0]]

        sections = []
        # Shuffle categories to avoid positional overfittings
        cat_list = ["personal", "academia", "education", "hobbies", "skills", "work_experience"]
        shuffle(cat_list)
        for category in cat_list:
            with open(f"./prompt_results/{category}.json", "r") as fp:
                cat2json[category] = json.load(fp)
        # print(cat2json["education"])
        
            filtered_jsons = [single_json for single_json in cat2json[category]["prompt_results"] 
                            if single_json["grad_year"] in [grad_year, ""] and 
                            single_json["position"] in [position, ""]]

            if len(filtered_jsons) > 0:
                # Randomly (10% probability) set the text to 0 for better invariance
                if np.random.binomial(n=1, size=1, p=0.1).tolist()[0] == 0:
                    section = filtered_jsons[np.random.choice(len(filtered_jsons), 1).tolist()[0]]["response"]
                else:
                    section = ""
                CV_dict[category] = section
                sections += [section]
            else:
                CV_dict[category] = ""
                print(f"No {category} found")

        # TODO: sample the different separators \n or \n\n etc.
        final_CV = separator.join(sections)
        CV_dict["overall"] = final_CV
        CV_list += [CV_dict]

    print(len(CV_list))
    with open(f"./prompt_results/complete_CVs_very_large2024.json", "w", encoding="utf-8") as fp:
        json.dump({"data": CV_list}, fp, ensure_ascii=False, indent=4)