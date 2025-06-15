import os
import sys
import json
import logging
from tqdm import tqdm
import openai
from baseline_util import *
from my_util import *
from multiprocessing import Process
import pandas as pd

# --- Configuration ---
OPENAI_MODEL = ""
OUTPUT_BASE_DIR = "../prediction/"
MAX_CODE_LENGTH = 1000000
BASE_URL = ""
OPENAI_API_KEY = ""

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)


formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 初始化 OpenAI 客户端
client = openai.OpenAI(
    base_url=BASE_URL,
    api_key=OPENAI_API_KEY,

)

def call_openai_model(file_id: str, filename: str, code_content: str, current_logger: logging.Logger) -> dict:
    """
 Call the OpenAI API to predict whether a code file contains a defect.
Includes robust error handling and default predictions on failure.

Parameters:
file_id (str): Unique ID of the file.
filename (str): Name of the code file.
code_content (str): Full contents of the code file.
current_logger (logging.Logger): Logger instance for the current process.

Returns:
dict: A dictionary containing the prediction results in the specified JSON format,
and any error information. If the API call fails or the response is malformed, a default no-defect result is returned.
    """


    prediction_data = {
        "id": file_id,
        "filename": filename,
        "defective": False,
        "error_type": None,
        "error_message": None,
        "raw_response": ""
    }

    original_code_length = len(code_content)


    if original_code_length > MAX_CODE_LENGTH:
        prediction_data['error_type'] = "CODE_LENGTH_EXCEEDED"
        prediction_data['error_message'] = (
            f"Code length ({original_code_length} characters) exceeds "
            f"MAX_CODE_LENGTH ({MAX_CODE_LENGTH} characters). File skipped."
        )
        current_logger.warning(
            f"[{filename}] Warning: {prediction_data['error_message']}"
        )
        return prediction_data


    messages = [
        {"role": "user",
         "content": "You are an expert software quality assurance engineer. Your task is to analyze the provided code/text file and identify any potential software defects, bugs, or vulnerabilities.Respond only with a JSON object."
                    f"""
        Analyze the following code from file '{filename}' and determine if it contains any defects.
        Code:
        ```
        {code_content}
        ```
        Your response MUST be a JSON object with the following structure:
        {{
            "id": "{file_id}",
            "filename": "{filename}",
            "defective": <true_or_false>
        }}
        The 'defective' field must be a boolean (true or false). Do not include any other text or explanation.
        """}
    ]

    response_text = ""
    clean_response_text = ""

    try:

        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            extra_body={"enable_thinking": False}
        )
        response_text = completion.choices[0].message.content
        prediction_data["raw_response"] = response_text
        current_logger.info(f"[{filename}] Raw API response: {response_text}")


        clean_response_text = response_text.strip()
        if clean_response_text.startswith("```json"):
            clean_response_text = clean_response_text[len("```json"):].strip()
        if clean_response_text.endswith("```"):
            clean_response_text = clean_response_text[:-len("```")].strip()


        parsed_json = json.loads(clean_response_text)


        if not isinstance(parsed_json.get("defective"), bool):
            current_logger.warning(
                f"Warning: 'defective' field in response for {filename} is not a boolean. "
                f"Received: {parsed_json.get('defective')}. Defaulting to False."
            )
            prediction_data['defective'] = False
        else:
            prediction_data['defective'] = parsed_json['defective']

    except openai.APIError as e:

        prediction_data['error_type'] = "API_ERROR"
        prediction_data['error_message'] = str(e)
        current_logger.error(f"OpenAI API Error for {filename}: {e}")
    except json.JSONDecodeError as e:

        prediction_data['error_type'] = "JSON_DECODE_ERROR"
        prediction_data['error_message'] = str(e)
        current_logger.error(
            f"JSON Decode Error for {filename}: {e}. "
            f"Original Response: '{response_text}'. Cleaned Response: '{clean_response_text}'"
        )
    except Exception as e:

        prediction_data['error_type'] = "UNEXPECTED_ERROR"
        prediction_data['error_message'] = str(e)
        current_logger.error(f"An unexpected error occurred for {filename}: {e}")

    return prediction_data


def predict_model_for_dataset(dataset_name: str):
    """
Orchestrate the prediction process for a given dataset.
Handles logging, data loading, API calls, and saving results/errors.
    """


    local_logger = logging.getLogger(f"prediction_logger_{dataset_name}")
    local_logger.setLevel(logging.INFO)

    local_logger.propagate = False

    log_file_name = f"prediction_process_{dataset_name}.log"
    local_log_file_path = os.path.join(OUTPUT_BASE_DIR, log_file_name)

    if local_logger.handlers:
        for handler in local_logger.handlers[:]:
            local_logger.removeHandler(handler)


    file_handler = logging.FileHandler(local_log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    local_logger.addHandler(file_handler)


    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    local_logger.addHandler(console_handler)

    local_logger.info(f"Starting prediction for dataset: {dataset_name}")


    if dataset_name not in all_eval_releases:
        local_logger.error(f"Error: Dataset '{dataset_name}' not found in all_eval_releases. Please check configuration.")
        return


    eval_rels = all_eval_releases[dataset_name][1:]


    error_records_for_dataset = []
    all_predictions_for_dataset = []

    prediction_id = 0
    for rel in eval_rels:
        local_logger.info(f"\nProcessing release: {rel}")
        row_list = []
        test_df = get_df(rel, is_baseline=True)


        for filename, df in tqdm(test_df.groupby('filename'), desc=f"[Predicting for {rel}]"):
            file_label = bool(df['file-label'].unique()[0])
            code = list(df['code_line'])
            code_str = get_code_str(code, True)
            prediction_id += 1


            prediction_result = call_openai_model(str(prediction_id), filename, code_str, local_logger)


            if prediction_result.get("error_type"):
                error_records_for_dataset.append({
                    "id": prediction_result["id"],
                    "filename": prediction_result["filename"],
                    "error_type": prediction_result["error_type"],
                    "error_message": prediction_result["error_message"],
                    "raw_response": prediction_result["raw_response"]
                })

            prediction_result['true_defective'] = file_label
            row_list.append(prediction_result)

        predictions_df = pd.DataFrame(row_list)
        output_csv_path = os.path.join(OUTPUT_BASE_DIR, f"{dataset_name}_{rel}_predictions.csv")
        predictions_df.to_csv(output_csv_path, index=False)
        local_logger.info(f"Predictions for {rel} saved to {output_csv_path}")


        all_predictions_for_dataset.extend(row_list)



    if error_records_for_dataset:
        errors_df = pd.DataFrame(error_records_for_dataset)
        error_csv_path = os.path.join(OUTPUT_BASE_DIR, f"{dataset_name}_errors.csv")
        errors_df.to_csv(error_csv_path, index=False)
        local_logger.warning(f"Error records for {dataset_name} saved to {error_csv_path}")
    else:
        local_logger.info(f"No errors recorded for dataset: {dataset_name}")

    local_logger.info(f"Finished prediction for dataset: {dataset_name}")


if __name__ == '__main__':

    datasets_to_run = ["activemq", "camel", "derby", "groovy", "hbase", "hive", "jruby", "lucene", "wicket"]

    processes = []



    for ds_name in datasets_to_run:
        p = Process(target=predict_model_for_dataset, args=(ds_name,))
        processes.append(p)
        p.start()


    for p in processes:
        p.join()


    print(f"所有日志文件和预测结果 CSV 文件都已保存到 '{os.path.abspath(OUTPUT_BASE_DIR)}' 目录中。")

