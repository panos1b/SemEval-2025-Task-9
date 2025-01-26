import pandas as pd
from sklearn.metrics import f1_score

def compute_score(hazards_true, products_true, hazards_pred, products_pred):
    # Compute F1 score for hazards
    f1_hazards = f1_score(hazards_true, hazards_pred, average='macro')
    
    # Compute F1 score for products where hazards match
    mask = hazards_pred == hazards_true
    f1_products = f1_score(products_true[mask], products_pred[mask], average='macro')
    
    # Return the average of the two F1 scores
    return (f1_hazards + f1_products) / 2

# Load the ground truth and submission files
df_true = pd.read_csv('incidents_set.csv', index_col=0)
df_pred = pd.read_csv('submission.csv', index_col=0)

# Rename columns to avoid conflicts during merge
df_true = df_true.rename(columns={
    'hazard-category': 'hazard_true',
    'product-category': 'product_true'
})
df_pred = df_pred.rename(columns={
    'hazard-category': 'hazard_pred',
    'product-category': 'product_pred'
})

# Merge dataframes on index to align true and predicted values
df_merged = df_true.join(df_pred, how='inner')

# Extract the aligned true and predicted values
hazards_true = df_merged['hazard_true']
products_true = df_merged['product_true']
hazards_pred = df_merged['hazard_pred']
products_pred = df_merged['product_pred']

# Calculate and print the final score
final_score = compute_score(hazards_true, products_true, hazards_pred, products_pred)
print(f"Final Evaluation Score: {final_score:.4f}")