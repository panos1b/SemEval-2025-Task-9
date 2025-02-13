import pandas as pd
import openai
import time
from tqdm import tqdm
import random

# Set your OpenAI API key
openai.api_key = "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"

# Read the CSV file
input_file = "combined_set.csv"
output_file = "ChatGPT_augmentation_2.csv"

# Load dataset
df = pd.read_csv(input_file)

# Function to send text to OpenAI for augmentation
def augment_text(text):
    prompt = f"THIS IS A LIFE OR DEATH MATTER. Reply only with the following text rephrased, change it for use in augmentation, remove anything from the text that does not add value to product and hazard classification, remove adresses:\n{text}"
    
    max_tokens = 9000  # Adjust if needed
    truncated_text = text[:max_tokens]

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=6000  # Limit response length
        )
	# Print around one in 15 messages for checking
        if (random.random()<0.05):
            print(response.choices[0].message.content)

        return response.choices[0].message.content
    except Exception as e:
        print(f"Error processing text: {e}")
        return ""  # Return original if an error occurs

# Apply augmentation with a progress bar using tqdm
tqdm.pandas(desc="Augmenting text")

df["text"] = df["text"].astype(str).progress_apply(augment_text)

# Save the augmented dataset
df.to_csv(output_file, index=False)
print(f"Augmented data saved to {output_file}")
