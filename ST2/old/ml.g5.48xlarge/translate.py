import os
import pandas as pd



from sklearn.model_selection import train_test_split

from deep_translator import GoogleTranslator


def back_translate(text, source_lang="en", target_lang="es"):
    if len(text) > 4999:
        text = text[:4999]  # Truncate to avoid exceeding the limit
    translated = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    if len(translated) > 4999:
        translated = translated[:4999]
    back_translated = GoogleTranslator(source=target_lang, target=source_lang).translate(translated)
    return back_translated


# Params
MODEL = 'FacebookAI/roberta-base'
MAX_LEN = 512
BATCH_SIZE = 90
EPOCHS = 60

# --- Data Loading and Preprocessing ---

# Read the CSV file
combined_df = pd.read_csv('combined_set.csv')

# Create a new column that combines 'product' and 'hazard'
combined_df['stratify_col'] = combined_df['product'].astype(str) + "_" + combined_df['hazard'].astype(str)

# Filter out unique occurrences in 'product' and 'hazard'
combined_df = combined_df[
    combined_df.duplicated(subset=['product'], keep=False) &
    combined_df.duplicated(subset=['hazard'], keep=False)
]

combined_df = combined_df[
    combined_df.duplicated(subset=['stratify_col'], keep=False)
]

# Perform stratified split
train_df, val_df = train_test_split(
    combined_df,
    test_size=0.3,
    random_state=69,
    stratify=combined_df['stratify_col']  # Stratify on combined column
)


if os.path.exists("text_augmented.csv"):
    train_df = pd.read_csv("text_augmented.csv")
    print("File loaded successfully.")
else:
    augmented_texts = train_df['text'].apply(lambda x: back_translate(x))

    new_rows = train_df.copy()
    new_rows['text'] = augmented_texts

    train_df = pd.concat([train_df, new_rows], ignore_index=True)

    train_df.to_csv("text_augmented.csv", index=False)

