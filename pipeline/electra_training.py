from dotenv import load_dotenv
import os
from transformers import ElectraForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
from datasets import Dataset

load_dotenv()

DATASET_PATH = os.getenv("DATASET_PATH")
MODEL_NAME = "Seznam/small-e-czech"

lb = LabelBinarizer()

df = pd.read_csv(DATASET_PATH)
df['label'] = lb.fit_transform(df['category'])

from transformers import ElectraTokenizer

tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME).to("cuda")
model = ElectraForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(df['label'].unique())
)

def tokenize(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

dataset = Dataset.from_pandas(df)
train_dataset = dataset.map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir="./results"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()