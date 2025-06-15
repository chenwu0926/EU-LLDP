import os
import json
import openai
import pandas as pd
from baseline_util import *
from my_util import *
from tqdm import tqdm

OPENAI_MODEL = ""

MAX_INPUT_LENGTH = 100000
CHUNK_SIZE = 95000
CHUNK_OVERLAP = 2000

BASE_URL = ""
OPENAI_API_KEY = ""


client = openai.OpenAI(
        base_url=BASE_URL,
        api_key=OPENAI_API_KEY,
    )



def _call_openai_for_chunk(content_chunk: str, filename: str, chunk_info: str) -> dict:

    print(f"  - Analyzing {chunk_info} for file '{filename}'...")

    default_response = {"defective": False, "error": "An error occurred"}

    prompt = (
        "You are an expert software quality assurance engineer. Your task is to analyze the provided code "
        "and identify any potential software defects, bugs, or vulnerabilities. Respond only with a JSON object."
        f"""
        Analyze the following code from file '{filename}'. This is a segment of a larger file: {chunk_info}.
        Code Segment:
        ```
        {content_chunk}
        ```
        Your response MUST be a JSON object with the following structure:
        {{
            "defective": <true_or_false>
        }}
        The 'defective' field must be a boolean (true or false). Do not include any other text or explanation.
        """
    )

    messages = [{"role": "user", "content": prompt}]

    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            extra_body={"response_format": {"type": "json_object"}}
        )
        response_text = completion.choices[0].message.content


        clean_response_text = response_text.strip()
        if clean_response_text.startswith("```json"):
            clean_response_text = clean_response_text[len("```json"):].strip()
        if clean_response_text.endswith("```"):
            clean_response_text = clean_response_text[:-len("```")].strip()

        parsed_json = json.loads(clean_response_text)

        if not isinstance(parsed_json.get("defective"), bool):
            print(f"  - Warning: 'defective' field is not a boolean. Defaulting to False. Response: {response_text}")
            parsed_json['defective'] = False

        return parsed_json

    except json.JSONDecodeError as e:
        print(f"  - Error: Failed to decode JSON from API response for {filename}. Error: {e}")
        print(f"  - Raw Response: {response_text}")
        return default_response
    except Exception as e:
        print(f"  - Error: An unexpected error occurred during API call for {filename}. Error: {e}")
        return default_response


def predict_defects_in_code(filename: str, code_content: str) -> bool:

    if len(code_content) <= MAX_INPUT_LENGTH:
        print(f"'{filename}' is within the size limit. Performing single analysis.")
        result = _call_openai_for_chunk(code_content,  filename, "Single Block")
        return result.get("defective", False)


    else:
        print(
            f"'{filename}' exceeds size limit ({len(code_content)} > {MAX_INPUT_LENGTH}). Starting chunk-based analysis.")
        chunks = []

        for i in range(0, len(code_content), CHUNK_SIZE - CHUNK_OVERLAP):
            chunks.append(code_content[i:i + CHUNK_SIZE])

        total_chunks = len(chunks)
        print(f"Code was split into {total_chunks} chunks.")

        for i, chunk in enumerate(chunks):
            chunk_info = f"Chunk {i + 1} of {total_chunks}"
            result = _call_openai_for_chunk(chunk, filename, chunk_info)

            if result.get("defective", False):
                print(f"  -> Defect found in {chunk_info}. Marking file '{filename}' as defective.")
                return True

        print(f"  -> No defects found in any chunk. Marking file '{filename}' as non-defective.")
        return False


if __name__ == '__main__':
    target_dataset_name = "wicket"
    target_file_id = [120,259,331,343,498,882,1283]
    eval_rels = all_eval_releases[target_dataset_name][1:]
    all_data_df = pd.DataFrame()
    prediction_id = 0
    for rel in eval_rels:
        row_list = []
        test_df = get_df(rel, is_baseline=True)
        for filename, df in tqdm(test_df.groupby('filename'), desc=f"[Predicting for {rel}]"):
            prediction_id += 1
            if prediction_id in target_file_id:
                file_label = bool(df['file-label'].unique()[0])
                code = list(df['code_line'])
                code_str = get_code_str(code, True)
                predicted_defect = predict_defects_in_code( filename, code_str)
                print(f"             File: {filename}")
                print(f" Predicted Status: {'Defective' if predicted_defect else 'Not Defective'}")

