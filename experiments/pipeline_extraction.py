from dotenv import load_dotenv
import torch
from transformers import ElectraTokenizer, ElectraForSequenceClassification
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from dotenv import load_dotenv
import os
import xgboost as xgb

load_dotenv()

MODEL_DICT = {
    "anger": "weights/anger",
    "clickbait": "weights/clickbait",
    "hate_speech": "weights/hate_speech",
    "political bias": "weights/political_bias",
    "stereotypization": "weights/stereotypization",
    "seriousness": "weights/seriousness",
}

IS_XGBOOST_PROCESSED = os.getenv('IS_XGBOOST') == 'true'
XGBOOST_PATH = os.getenv('XGBOOST_PATH')

DATASET_PATH = os.getenv("DATASET_PATH")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = ElectraTokenizer.from_pretrained("Seznam/small-e-czech")

df = pd.read_csv(DATASET_PATH)
texts = df['text'].tolist()
encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'])

for model_name in MODEL_DICT.keys():

    model = ElectraForSequenceClassification.from_pretrained(MODEL_DICT[model_name]).to(device)

    model.eval()

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            predictions.extend(probs[:, 0])
    
    df[f'{model_name}_pred'] = predictions

if IS_XGBOOST_PROCESSED:
    model = xgb.XGBClassifier()
    model.load_model(XGBOOST_PATH)

    feature_columns = [col for col in df.columns if col.endswith('_pred')]
    dtest = xgb.DMatrix(df[feature_columns])

    df["xgboost"] = model.predict(dtest)

df.to_csv("results.csv", index=False)