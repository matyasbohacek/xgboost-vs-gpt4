import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

from dotenv import load_dotenv

load_dotenv()

DATASET_PATH = os.getenv('API_KEY')

df = pd.read_csv(DATASET_PATH)

feature_columns = [col for col in df.columns if col.endswith('_pred') or col == 'category']
df = df[feature_columns]

df = df.sample(frac=1, random_state=42)

X = df.drop(columns=['category'])
y = df['category']

dtrain = xgb.DMatrix(X, label=y)
bst = xgb.train(dtrain)

bst.save_model('xgboost_model.json')