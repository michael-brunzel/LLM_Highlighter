# App for displaying the marked texts that belong to the differen topics

import gradio as gr
import requests
import os
import json
import re
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.environ.get("API_TOKEN", None)
MODEL_URL = os.environ.get("MODEL_URL", None)


def predict(CV_text: str):
    resp = requests.post(
        MODEL_URL,
        json={"data": {"inputs": CV_text, "parameters": {"max_new_tokens": 128, "do_sample": False}}},
        headers={"Authorization": f"Bearer {API_TOKEN}"},
        cookies=None,
        timeout=10,
    )
    payload = resp.json()
    output =  payload["body"]

    output = re.sub("\'", "\"", output)
    output = output.replace("\\n", "")
    new_dict = json.loads(output)
    entities = []

    for key, _ in new_dict.items():
        start = new_dict[key]["s"]
        end = new_dict[key]["e"]
        if start != "":
            single_sequence = start + f"{start}".join(CV_text.split(start)[1:]).split(end)[0] + end
            overall_start = int(CV_text.find(single_sequence))
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
    
    return {
        "text": CV_text,
        "entities": entities
    }

# css=".input_text textarea {color: white;}"  # Change text color to red
# css="#textbox_id span {color: white} #textbox_id div {color: white}"
with gr.Blocks() as demo:
    with gr.Column():
        gr.Interface(
            predict,
            [
                gr.Textbox(
                    label="Text 1",
                    info="Initial text",
                    lines=3,
                    value="The quick brown fox jumped over the lazy dogs.",
                ),
            ],
            gr.HighlightedText(
                label="LLM Textmarker",
                elem_id="textbox_id",
            ),
            inline=False,
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")

