import pandas as pd
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import ollama

# File paths
input_file = "combined_set.csv"
output_file = "ChatGPT_augmentation.csv"

# Load dataset
df = pd.read_csv(input_file)

# Function to send text to local Ollama instance for augmentation
def augment_text(text, counter):
    prompt = f"THIS IS A LIFE OR DEATH MATTER. Reply only with the following text rephrased, change it for use in augmentation, remove anything from the text that does not add value:\n{text}"
    
    # Use the Ollama Python API to send the prompt
    try:
        response = ollama.chat(
            model="deepseek-r1:8b",
            messages=[{"role": "user", "content": prompt}],
        )
        print(f"Processing text {counter + 1}")  # Print the current counter
	if counter % 20 == 0:
            print(response["message"]["content"])
        return response["message"]["content"]
    
    except Exception as e:
        print(f"Error processing text {counter + 1}: {e}")
        return ""  # Return original if an error occurs

# Threaded function for augmenting text in parallel
def augment_parallel(df, num_threads=192):
    augmented_texts = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for counter, augmented_text in enumerate(executor.map(augment_text, df["text"].astype(str), range(len(df)))):
            augmented_texts.append(augmented_text)
    return augmented_texts

# Apply augmentation with a progress bar using tqdm (without OpenAI, now using local server)
tqdm.pandas(desc="Augmenting text")

# Perform augmentation in parallel to speed things up (use hyperthreading)
print("Start")
df["text"] = augment_parallel(df)

# Save the augmented dataset
df.to_csv(output_file, index=False)
print(f"Augmented data saved to {output_file}")
