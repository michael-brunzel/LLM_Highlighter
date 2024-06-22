import copy
import json
import re
from typing import List, Tuple
import argilla as rg

def determine_ner_entities(llm_output: str, cv_text: str) -> List[dict]:
    """
    Determine the actual text outputs and their string positions.
    
    Args:
        llm_output (str): The raw output of the LLM or the actual label
        cv_text (str): The full CV
    """
    new_dict = json.loads(re.sub("\'", "\"", llm_output))
    entities = []
    for key, _ in new_dict.items():
        start = new_dict[key]["s"]
        end = new_dict[key]["e"]
        if start != "":
            single_sequence = start + f"{start}".join(cv_text.split(start)[1:]).split(end)[0] + end
            overall_start = int(cv_text.find(single_sequence))
            # If text sequence couldn't be found -> error in the prediction
            if overall_start == -1:
                continue
            entities += [{
                "entity": key,
                "word": single_sequence,
                "start": overall_start, # start position in the whole string sequence
                "end": overall_start+len(single_sequence), # end position in the whole string sequence
                "score": 0,
            }]
    
    return entities


def find_all(a_str, sub):
    """
    This function finds all occurences of a sub-string within a string
    
    Args:
        a_str (str): The complete string
        sub (str): the substring which should be found
    """
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches

        
def split_up_entities(topic_text: str, topic_start: int, topic: str) -> List[Tuple[str, int, int]]:
    """
    Compute the word locations for all words in the topic text
    
    Args:
        topic_text (str): The text for a particular topic
        topic_start (int): The start index of the topic text within the whole CV
        topic (str): The entity name of the topic
    """
    text_splitted = topic_text.split() # the whitespace is removed --> count the positions in between
    collected_splits = []
    for idx, split in enumerate(text_splitted):
        if idx > 0:
            # previous_text = " ".join(text_splitted[:idx])
            found_indices = find_all(topic_text, split)
            correct_idx = [i for i in found_indices if i > (new_end - topic_start)]
            new_start = correct_idx[0] + topic_start
            new_end = new_start + len(split)
        else:
            new_start = topic_text.find(split) + topic_start
            new_end = len(split) + new_start #

        # print(pe_text[new_start:new_end+1])
        collected_splits += [(topic, new_start, new_end)]

    return collected_splits


def compute_argilla_entities(preds, labels, texts, prediction_agent: str):
    records = []
    for idx in range(len(texts)):
        print(idx)
        label_entities = determine_ner_entities(labels[idx], texts[idx])
        pred_entities = determine_ner_entities(preds[idx], texts[idx])

        label_splits = []
        for ent in label_entities:
            label_splits += split_up_entities(ent["word"], ent["start"], ent["entity"])

        pred_splits = []
        for ent in pred_entities:
            pred_splits += split_up_entities(ent["word"], ent["start"], ent["entity"])
    
        # Argilla TokenClassificationRecord list
        records.append(
            rg.TokenClassificationRecord(
                text=texts[idx],
                tokens=texts[idx].split(), #(" "),
                prediction=pred_splits,
                annotation=label_splits,
                prediction_agent=prediction_agent,
            )
        )
    token_dataset = rg.DatasetForTokenClassification(records)
    return token_dataset


def token2positions(text: str, annotations: List[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
    """
    Compute the word locations for all words in the text and map the unallocated words (no category yet) as category 'none'
    
    Args:
        text (str): The whole CV text
        annotations (list): A list of tuples where each tuple contains the category and the start and end position of a token

    Return:
        (list): The list of annotations which were extended by the tokens that belong to the category 'none'
    """
    mod_annotations = copy.deepcopy(annotations)
    mod_annotations = [tup[0:3] for tup in mod_annotations]
    text_splitted = text.split() # the whitespace is removed --> count the positions in between
    for idx, split in enumerate(text_splitted):
        if idx > 0:
            found_indices = find_all(text, split)
            correct_idx = [i for i in found_indices if i > new_end]
            new_start = correct_idx[0]
            new_end = new_start + len(split)
        else:
            new_start = text.find(split)
            new_end = len(split) + new_start

        filtered_annotations = [anno for anno in mod_annotations if anno[1] == new_start]
        if len(filtered_annotations) == 0:
            mod_annotations += [("none", new_start, new_end)]
    
    # TODO: filter out categories which are ambiguous (same range for different categories...) --> for simplicity: filter out the first category...
    ranges = [tup[1:3] for tup in mod_annotations]
    unique_ranges = list(set(ranges))
    amb_ranges = []
    for ra in unique_ranges:
        if len([r for r in ranges if ra == r]) > 1:
            amb_ranges += [ra]
    amb_annotations = [amb for amb in mod_annotations if amb[1:3] in amb_ranges]
    amb_cats = list(set([amb[0] for amb in amb_annotations]))
    amb_annotations = [amb for amb in amb_annotations if amb[0] == amb_cats[0]]
    mod_annotations = [anno for anno in mod_annotations if anno not in amb_annotations]

    return mod_annotations


def compute_sequences(dataset):
    """
    Compute the words for the label sequence and the prediction sequence respectively.
    
    Args:
        dataset: A loaded argilla dataset
    """
    overall_labels = []
    overall_predictions = []
    overall_pred_triples = []
    overall_label_triples = []
    for idx in range(len(dataset)):
        label_triples = token2positions(dataset[idx].text, dataset[idx].annotation)
        prediction_triples = token2positions(dataset[idx].text, dataset[idx].prediction)

        label_triples = sorted(label_triples, key= lambda x: x[1], reverse=False)
        prediction_triples = sorted(prediction_triples, key= lambda x: x[1], reverse=False)
        labels, start, end = zip(*label_triples)
        # print(prediction_triples)
        predictions, start, end = zip(*prediction_triples)
        
        overall_labels += [labels]
        overall_predictions += [predictions]
        overall_pred_triples += [prediction_triples]
        overall_label_triples += [label_triples]
    return overall_labels, overall_predictions, overall_pred_triples, overall_label_triples