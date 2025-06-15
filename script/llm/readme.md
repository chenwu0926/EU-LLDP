# Evaluating Large Language Models for Software Defect Prediction

## 1. Overview

This project investigates the efficacy of modern Large Language Models (LLMs) in the domain of software defect prediction. We conduct a comparative analysis between several prominent LLMs and our proposed method, **EU-LLDP**, to benchmark their performance on this specialized software engineering task.


## 2. Workflow and Usage

The experimental workflow is divided into three main stages: data preprocessing, defect prediction using the LLM, and special handling for large files.

### Step 1: Data Preprocessing

First, process the raw dataset to create a standardized format for the prediction task.

```bash
python preprocess_data.py
```

This script (`preprocess_data.py`) reads the original file-level and line-level datasets. For each Java source file, it extracts the source code (`SRC`), class label (`Bug`), and line-level defect information. It then structures this data into a `.csv` file for each release, containing columns for `filename`, `code_line`, `line_number`, `file-label`, and `line-label`.

### Step 2: Defect Prediction with LLMs

Next, run the main prediction script to send the processed code to the LLM for analysis.

```bash
python llm.py
```

The `llm.py` script orchestrates the prediction process. For each code file in the evaluation set, it performs the following steps:
1.  **Constructs a Prompt**: The script uses a specifically engineered prompt to instruct the LLM to act as a software quality assurance expert and identify potential defects. The prompt explicitly requests a JSON object as output to ensure structured and parsable responses.
```commandline
prompt 

You are an expert software quality assurance engineer. Your task is to analyze the provided code/text file and identify any potential software defects, bugs, or vulnerabilities.Respond only with a JSON object."
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
```
2.  **API Call Configuration**: The model is invoked using the official recommended configurations. Notably, advanced reasoning features like "thinking mode" are disabled (`enable_thinking: False`) to rely solely on the model's foundational knowledge. Most other parameters are set to their default values.
3.  **Parses Response**: It parses the JSON output from the LLM to determine if a file is predicted as `defective` (true) or `non-defective` (false).
4.  **Saves Results**: The predictions, along with the ground truth labels, are saved to a `.csv` file for subsequent analysis.

### Step 3: Handling Large Code Files

For source code files that exceed the LLM's maximum input token limit, a separate script is used to perform a chunk-based analysis.

```bash
python process_error.py
```

The `process_error.py` script implements a chunking strategy:
1.  The large source file is divided into smaller, overlapping segments (chunks).
2.  Each chunk is sent to the LLM individually for defect analysis using the same prompting strategy.
3.  If **any single chunk** is classified as defective by the model, the entire file is considered defective. The file is only classified as non-defective if all of its chunks are predicted to be non-defective.

## 3. Experimental Analysis (RQ2)

### Research Question
**RQ2: How does EU-LLDP perform compared to large language models?**

### Methodology

We conducted a comparative analysis of four prominent LLMs: **GPT-4.1 Mini**, **Gemini 2.5 Flash Preview 05-20**, **Llama 4 Maverick**, and **Qwen3**. Given the closed-source nature of most models, we adopted a **zero-shot learning** approach, where the models perform predictions based on a general prompt without any prior fine-tuning on domain-specific data. Performance was measured using Balanced Accuracy, Matthews Correlation Coefficient (MCC), Precision, and Recall.

### Results

The performance of the LLMs compared to the baseline (DeepLineDP) and our proposed EU-LLDP method is summarized below.

| Model                       | Balanced Accuracy | MCC     | Precision | Recall  |
| --------------------------- | ----------------- | ------- | --------- | ------- |
| DeepLineDP                  | 0.628             | 0.126   | 0.990     | 0.404   |
| GPT-4.1 Mini                | 0.502             | 0.013   | 0.043     | 0.976   |
| Llama 4 Maverick            | 0.498             | -0.002  | 0.042     | 0.506   |
| Gemini 2.5 Flash Preview    | 0.498             | -0.003  | 0.038     | 0.277   |
| Qwen3                       | 0.539             | 0.030   | 0.044     | 0.468   |
| **EU-LLDP (DP)** | **0.690** | **0.174** | **0.988** | **0.562** |

### Conclusion

The experimental results clearly indicate that our **EU-LLDP** method significantly outperforms the evaluated general-purpose LLMs in a zero-shot setting. While models like GPT-4.1 Mini achieve very high Recall, their Precision is extremely low, suggesting a tendency to classify non-defective files as defective. The negative MCC scores for Llama 4 and Gemini 2.5 indicate their performance is worse than random guessing.

These findings establish a lower-bound performance for LLMs on this task and highlight that, without specialized training or fine-tuning, their utility for reliable defect prediction is limited. Furthermore, the high computational and financial costs associated with invoking LLM APIs make EU-LLDP a more cost-effective and superior solution.

## 4. Threats to Validity

The validity of our findings is subject to the following threats:

* **LLM Stochasticity**: The non-deterministic nature of LLMs means that repeated queries with the same input may yield different results, potentially affecting the reliability and reproducibility of the predictions.
* **Model Hallucination**: LLMs can generate plausible but factually incorrect information ("hallucinate"). In this context, a model might identify a "defect" that does not logically exist or misinterpret a code pattern, leading to false positives.
* **Prompt Engineering Sensitivity**: The performance of LLMs is highly sensitive to the structure and content of the input prompt. Different prompt designs could lead to different results, and our engineered prompt may not be universally optimal.

## 5. Future Work

Future research will focus on enhancing LLM performance in this domain. Key directions include:

* **Fine-Tuning**: Fine-tuning LLMs on a large, high-quality dataset of software defects and corresponding code. This would equip the model with specialized knowledge to better distinguish between defective and non-defective patterns.
* **Advanced Prompting Techniques**: Exploring more sophisticated prompting strategies, such as Chain-of-Thought (CoT) or few-shot prompting, to guide the model's reasoning process more effectively.
* **Hybrid Approaches**: Developing hybrid models that combine the pattern-recognition capabilities of LLMs with traditional static analysis tools or specialized defect prediction models like EU-LLDP.

