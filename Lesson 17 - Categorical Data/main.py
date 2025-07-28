# Import required libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Sample dataset with categorical data
data = {
    'Country': ['India', 'USA', 'UK', 'India', 'UK'],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
    'Purchased': ['No', 'Yes', 'No', 'No', 'Yes']
}

# Create DataFrame
df = pd.DataFrame(data)

# Display original data
print("Original Data:")
print(df)

# -------------------------
# LABEL ENCODING
# -------------------------

# Label Encoding 'Gender' and 'Purchased' columns
label_encoder_gender = LabelEncoder()
label_encoder_purchased = LabelEncoder()

df['Gender'] = label_encoder_gender.fit_transform(df['Gender'])
df['Purchased'] = label_encoder_purchased.fit_transform(df['Purchased'])

print("\nAfter Label Encoding:")
print(df)

# -------------------------
# ONE HOT ENCODING
# -------------------------

# One-hot encode the 'Country' column (create dummy variables)
# We use ColumnTransformer to apply OneHotEncoder to just one column

ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), ['Country'])
    ],
    remainder='passthrough'  # Keep other columns unchanged
)

df_encoded = ct.fit_transform(df)

# Convert the result into a DataFrame
df_encoded = pd.DataFrame(df_encoded)

print("\nAfter One-Hot Encoding:")
print(df_encoded)
