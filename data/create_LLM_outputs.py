"""
Add the output labels for the LLM training (unambiguous beginning and end of the section)
in order to get the necessary infos about the different section.
Load the existing data and find a good start and end; use json as format for the outputs
"""

import json
import re
from collections import OrderedDict

def determine_start_and_end(topic_string: str, overall_string: str) -> tuple:
    """
    This function computes minimal start- and end sequence for a specific topic.

    Args:
        topic_string (str): The string which contains the topic section from the CV
        overall_string (str): The complete CV
    
    Return:
        (tuple): The minimal start- and end sequence for each topic.
    """
    matches_before = []
    idx = 0
    topic_words = re.findall(r"\w+", topic_string)
    # topic_words = topic_string.split(" ")

    while True:
        # matched_words = re.findall(edu_words[idx], complete_string)
        # print(topic_words[idx])
        # print("".join(topic_string.split(topic_words[idx])[1:]))
        start_sequence = topic_string[:-len(topic_string.split(topic_words[idx], 1)[1])]
        # print(start_sequence)
        # check if there is already a match before the start of the section
        matches_before = re.findall(start_sequence, overall_string.split(topic_string)[0])
        # print(matches_before)
        if len(matches_before) == 0:
            break
        idx += 1

    # Code for determining the end sequence
    # only check that end_sequence is unique within the passage (is sufficient)
    end_idx = len(topic_words) -1
    while True:
        # The following code only makes sends if there is positive length match
        while len(topic_string.split(topic_words[end_idx])[-1]) == 0:
            end_idx -= 1
        end_sequence = topic_string[-len(topic_string.split(topic_words[end_idx])[-1]):]
        split_sequence = re.sub(r"[\(\)]", "", end_sequence)
        if split_sequence != "":
            matches_before = re.findall(split_sequence, topic_string.rsplit(split_sequence, 1)[0])
            if len(matches_before) == 0:
                break
        end_idx -= 1
    
    return start_sequence, end_sequence

if __name__ == "__main__":
    with open(f"./prompt_results/complete_CVs_very_large2024.json", "r", encoding="utf-8") as fp:
        CV_list = json.load(fp)["data"]
    
    for CV in CV_list:
        label_dict = {}
        for cat, text in CV.items():
            # print(cat)
            # Academia and Hobbies should be ignored for the learning task
            if cat in ["personal", "education", "work_experience", "skills"] and text != "":
                start, end = determine_start_and_end(text, CV["overall"])
                label_dict[cat[:2]] = {
                    "s": start,
                    "e": end
                }
            else:
                if cat in ["personal", "education", "work_experience", "skills"]:
                # Add empty entry to stick to general Json format
                    label_dict[cat[:2]] = {
                        "s": "",
                        "e": ""
                    }

        # Keep the order of the json schema in the final dictionary, since the order can be consistent for the LLM      
        final_dict = {}
        for key in ["pe", "ed", "wo", "sk"]:
            final_dict[key] = label_dict[key]
        CV["output"] = str(final_dict)

    with open(f"./prompt_results/complete_CVs_very_large_with_labels_new20244_bernoulli.json", "w", encoding="utf-8") as fp:
        json.dump({"data": CV_list}, fp, ensure_ascii=False, indent=4)