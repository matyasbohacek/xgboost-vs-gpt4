
from tqdm import tqdm
from datasets.verifee_ds import load_dataset
from gpt.inference import infer_article_gpt_4

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

dataset_path = ""
dataset_split = ""
output_path = ""
cap = 40

dataset = load_dataset(dataset_path, dataset_split)
small_sampled_dataset = dataset.groupby("label").sample(n=cap, random_state=1)

label_meta, explanation_meta, tokens_used_meta = [], [], []

for row_i, row in tqdm(small_sampled_dataset.iterrows()):
    label, explanation, tokens_used = infer_article_gpt_4(row)

    label_meta.append(label)
    explanation_meta.append(explanation)
    tokens_used_meta.append(tokens_used)

small_sampled_dataset["label"] = label_meta
small_sampled_dataset["explanation"] = explanation_meta
small_sampled_dataset["tokens"] = tokens_used_meta

small_sampled_dataset.to_csv(output_path, encoding="utf-8")
