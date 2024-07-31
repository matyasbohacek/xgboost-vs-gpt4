import deepl
import os
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('API_KEY')
file_to_translate = os.getenv("FILE_TO_TRANSLATE")

translator = deepl.Translator(api_key)

df = pd.read_csv(file_to_translate)

df["translated"] = [response.text for response in translator.translate_text(df["text"].to_list(), target_lang="CS")]

df.to_csv(f"{file_to_translate}_translated.csv", index = False)