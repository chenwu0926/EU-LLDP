# Supplementary Materials for "Estimating Uncertainty in Line-Level Defect Prediction via Perceptual Borderline Oversampling"

This repository contains the supplementary materials for the paper "Estimating Uncertainty in Line-Level Defect Prediction via Perceptual Borderline Oversampling". It includes the necessary datasets, source code, and instructions to replicate the experiments.

## Table of Contents

* [Datasets](#datasets)
* [File Structure](#file-structure)
* [Environment Setup](#environment-setup)
    * [Python Environment Setup](#python-environment-setup)
    * [R Environment Setup](#r-environment-setup)
* [Experiment Replication](#experiment-replication)
    * [Experimental Hyperparameters](#experimental-hyperparameters)
    * [Step 1: Data Preprocessing](#step-1-data-preprocessing)
    * [Step 2: Word2Vec Model Training](#step-2-word2vec-model-training)
    * [Step 3: Core EU-LLDP Model Training and Prediction](#step-3-core-eu-lldp-model-training-and-prediction)
    * [Step 4 (RQ1): Evaluating File-Level Baselines](#step-4-rq1-evaluating-file-level-baselines)
    * [Step 5 (RQ3): Evaluating Line-Level Baselines](#step-5-rq3-evaluating-line-level-baselines)
    * [Step 6 (RQ4): Comparison with Alternative Oversampling Methods](#step-6-rq4-comparison-with-alternative-oversampling-methods)
    * [Step 7 (RQ5): Ablation Study](#step-7-rq5-ablation-study)
    * [Step 8 (RQ6): Cross-Project Prediction and Robustness](#step-8-rq6-cross-project-prediction-and-robustness)
    * [Step 9: Statistical Significance Analysis](#step-9-statistical-significance-analysis)
* [[LLM] Training Details, Threats to Validity, and Future Work](./script/llm/readme.md)

---

## Datasets

The datasets are obtained from Wattanakriengkrai et al. and contain 32 software releases across 9 software projects. You can download the datasets used in our experiment from this [GitHub repository](https://github.com/awsm-research/line-level-defect-prediction).

* **File-level datasets** (`File-level` directory) contain the columns: `File`, `Bug`, `SRC`.
* **Line-level datasets** (`Line-level` directory) contain the columns: `File`, `Commit`, `Line_number`, `SRC`.

For each project, the oldest release is used for training, the subsequent release for validation, and all others for testing.

---

## File Structure

```
.
├── datasets/
│   ├── original/
│   ├── preprocessed_data/
│   ├── n_gram_data/
│   └── ErrorProne_data/
├── output/
│   ├── loss/
│   ├── model/
│   ├── prediction/
│   └── Word2Vec_model/
└── script/
    ├── RQ/
    ├── file-level-baseline/
    ├── line-level-baseline/
    ├── EU-LLDP_model.py
    ├── ... (other scripts)
    └── train_word2vec.py
```

* **`datasets/`**: Stores raw, preprocessed, and baseline-specific data.
* **`output/`**: Contains sub-directories for storing training/validation loss, trained models, predictions, and Word2Vec models.
* **`script/`**: Contains all Python and R scripts for preprocessing, training, prediction, and evaluation for the main model and all Research Questions.

---

## Environment Setup

### Python Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/chenwu0926/EU-LLDP.git](https://github.com/chenwu0926/EU-LLDP.git)
    ```
2.  **Download the dataset** from this [GitHub repository](https://github.com/awsm-research/line-level-defect-prediction) and place the contents into `./datasets/original/`.
3.  **Create the Conda environment:**
    ```bash
    conda env create -f requirements.yml
    conda activate EU-LLDP_env
    ```
4.  **Install PyTorch** by following the official instructions from this [link](https://pytorch.org/), as installation varies by OS and CUDA version.

### R Environment Setup

Install the following R packages: `tidyverse`, `gridExtra`, `ModelMetrics`, `caret`, `reshape2`, `pROC`, `effsize`, `ScottKnottESD`.

---

## Experiment Replication

### Experimental Hyperparameters

The following hyperparameters are used for training the EU-LLDP model:
* `batch_size`: 32
* `num_epochs`: 20
* `embed_dim` (word embedding size): 50
* `word_gcn_hidden_dim`: 128
* `sent_gcn_hidden_dim`: 128
* `dropout`: 0.2
* `lr` (learning rate): 0.001
* `weighted_graph`: `False`

### Step 1: Data Preprocessing

1.  **Prepare data for file-level models.** Output is stored in `./datasets/preprocessed_data`.
    ```bash
    python preprocess_data.py
    ```
2.  **Prepare data for line-level baselines.** Output is stored in `./datasets/ErrorProne_data/` and `./datasets/n_gram_data/`.
    ```bash
    python export_data_for_line_level_baseline.py
    ```

### Step 2: Word2Vec Model Training

Train Word2Vec models for each project. Replace `<DATASET_NAME>` with one of: `activemq`, `camel`, `derby`, `groovy`, `hbase`, `hive`, `jruby`, `lucene`, `wicket`.
```bash
python train_word2vec.py <DATASET_NAME>
```

### Step 3: Core EU-LLDP Model Training and Prediction

1.  **Train the EU-LLDP models.** Trained models are saved in `../output/model/EU-LLDP/<DATASET_NAME>/`.
    ```bash
    python train_model.py -dataset <DATASET_NAME>
    ```
2.  **Generate within-project predictions.** Output CSVs are stored in `../output/prediction/DeepLineDP/within-release/`.
    ```bash
    python generate_prediction.py -dataset <DATASET_NAME>
    ```

### Step 4 (RQ1): Evaluating File-Level Baselines

The four file-level baselines are `Bi-LSTM`, `CNN`, `DBN`, and `BoW`. To run them, navigate to the `./script/file-level-baseline/` directory.

1.  **Train the baseline models:**
    ```bash
    # Replace <DATASET_NAME> accordingly for each command
    python Bi-LSTM-baseline.py -data <DATASET_NAME> -train
    python CNN-baseline.py -data <DATASET_NAME> -train
    python DBN-baseline.py -data <DATASET_NAME> -train
    python BoW-baseline.py -data <DATASET_NAME> -train
    ```
2.  **Generate predictions from the baseline models:**
    ```bash
    # Replace <DATASET_NAME> accordingly for each command
    python Bi-LSTM-baseline.py -data <DATASET_NAME> -predict -target_epochs 6
    python CNN-baseline.py -data <DATASET_NAME> -predict -target_epochs 6
    python DBN-baseline.py -data <DATASET_NAME> -predict
    python BoW-baseline.py -data <DATASET_NAME> -predict
    ```

### Step 5 (RQ3): Evaluating Line-Level Baselines

1.  **N-gram**: Navigate to `/script/line-level-baseline/ngram/` and run the code in `n_gram.java`. Copy the resulting `/n_gram_result/` directory to the `/output/` directory.
2.  **ErrorProne**: Navigate to `/script/line-level-baseline/ErrorProne/` and run the `run_ErrorProne.ipynb` notebook. Copy the resulting `/ErrorProne_result/` directory to the `/output/` directory.

### Step 6 (RQ4): Comparison with Alternative Oversampling Methods

To compare EU-LLDP against other methods (`CVAE`, `ROS`, `BorderlineSMOTE`, `Adaptive Synthetic SMOTE`), navigate into each subdirectory within `script/RQ/RQ4/` and repeat the EU-LLDP training and prediction process from **Step 3**.

### Step 7 (RQ5): Ablation Study

For the ablation study (`CL`, `BORDERLINE`, `CL ADAROS`), navigate into each subdirectory within `script/RQ/RQ5/` and repeat the EU-LLDP training and prediction process from **Step 3**.

### Step 8 (RQ6): Cross-Project Prediction and Robustness

To evaluate the model's generalization ability, generate cross-project predictions. The output CSVs will be stored in `../output/prediction/DeepLineDP/cross-project/`.
```bash
python generate_prediction_cross_projects.py -dataset <DATASET_NAME>
```

### Step 9: Statistical Significance Analysis

To determine if the performance differences between EU-LLDP and other models are statistically significant, perform a paired t-test on the evaluation results. For guidance on using the `T.TEST` function, you can refer to the Microsoft Excel documentation [here](https://support.microsoft.com/en-us/office/t-test-function-d4e08ec3-c545-485f-962e-276f7cbed055).
