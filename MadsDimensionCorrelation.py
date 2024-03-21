import os
import librosa
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # Importing seaborn for enhanced visualizations
from sklearn.preprocessing import StandardScaler
import numpy as np

drumFolder = '500_Sounds'
features_list = []

for filename in os.listdir(drumFolder):
    if filename.endswith('.wav'):
        file_path = os.path.join(drumFolder, filename)
        y, sr = librosa.load(file_path)

        # Extract features and summarize as needed
        features = {
            'Filename': filename,
            'Spectral Centroid': librosa.feature.spectral_centroid(y=y, sr=sr).mean(),
            'Spectral Bandwidth': librosa.feature.spectral_bandwidth(y=y, sr=sr).mean(),
            'Spectral Rolloff': librosa.feature.spectral_rolloff(y=y, sr=sr).mean(),
            'Zero Crossing Rate': librosa.feature.zero_crossing_rate(y).mean(),
            'RMS Energy': librosa.feature.rms(y=y).mean(),
            'Spectral Flatness': librosa.feature.spectral_flatness(y=y).mean(),
            'Spectral Contrast': librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1).mean(),
            'Onset Envelope': librosa.onset.onset_strength(y=y, sr=sr).mean(),
        }
        features_list.append(features)

# Convert features_list to a DataFrame
features_dataframe = pd.DataFrame(features_list)

# Dropping 'Filename' for transformations and scaling
features_for_transformation = features_dataframe.drop('Filename', axis=1)
features_for_transformation += 1e-6  # Ensure positive values
log_transformed_features = np.log(features_for_transformation)

# Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(log_transformed_features)

# Convert scaled features back to DataFrame for correlation calculation
scaled_features_df = pd.DataFrame(scaled_features, columns=features_for_transformation.columns)

# Calculate the correlation matrix
correlation_matrix = scaled_features_df.corr()

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Audio Features')
plt.show()

# Generating pair plots for log-transformed features to see pairwise relationships
# It's more resource-intensive, so you might limit this to a subset of features if needed
log_transformed_features_df = pd.DataFrame(log_transformed_features, columns=features_for_transformation.columns)
sns.pairplot(log_transformed_features_df)
plt.show()
