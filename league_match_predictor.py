"""
League of Legends Match Predictor
---------------------------------
A PyTorch-based binary classification project that predicts
whether the blue team wins a League of Legends match.

Expected dataset:
high_diamond_ranked_10min.csv

The dataset should contain:
- blueWins (target column)
- Numerical match statistics for both teams
"""

import os
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DATASET_PATH = "high_diamond_ranked_10min.csv"

TEST_SIZE = 0.2
RANDOM_STATE = 42
LEARNING_RATE = 0.001
EPOCHS = 100


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset '{DATASET_PATH}' was not found. "
        "Download the League of Legends dataset and place it "
        "in the project folder."
    )

print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


# --------------------------------------------------
# 3. DATA PREPROCESSING
# --------------------------------------------------

TARGET_COLUMN = "blueWins"

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' was not found."
    )

# Remove non-numeric columns if present
df = df.select_dtypes(include=["number"])

# Remove missing values
df = df.dropna()

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

print("\nNumber of features:", X.shape[1])
print("Number of samples:", X.shape[0])


# --------------------------------------------------
# 4. TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


# --------------------------------------------------
# 5. FEATURE SCALING
# --------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --------------------------------------------------
# 6. CONVERT DATA TO PYTORCH TENSORS
# --------------------------------------------------

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train.values,
    dtype=torch.float32
).view(-1, 1)

y_test_tensor = torch.tensor(
    y_test.values,
    dtype=torch.float32
).view(-1, 1)


# --------------------------------------------------
# 7. DEFINE NEURAL NETWORK
# --------------------------------------------------

class MatchPredictor(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


input_size = X_train.shape[1]

model = MatchPredictor(input_size)

print("\nModel Architecture:")
print(model)


# --------------------------------------------------
# 8. LOSS FUNCTION AND OPTIMIZER
# --------------------------------------------------

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# 9. TRAIN THE MODEL
# --------------------------------------------------

print("\nTraining model...")

for epoch in range(EPOCHS):

    model.train()

    outputs = model(X_train_tensor)

    loss = criterion(
        outputs,
        y_train_tensor
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}], "
            f"Loss: {loss.item():.4f}"
        )


# --------------------------------------------------
# 10. MODEL EVALUATION
# --------------------------------------------------

model.eval()

with torch.no_grad():

    predictions = model(
        X_test_tensor
    )

    predicted_classes = (
        predictions >= 0.5
    ).int()

accuracy = accuracy_score(
    y_test_tensor.numpy(),
    predicted_classes.numpy()
)

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"Test Accuracy: {accuracy:.4f}")


# --------------------------------------------------
# 11. SAVE THE MODEL
# --------------------------------------------------

torch.save(
    model.state_dict(),
    "league_match_predictor.pth"
)

print(
    "\nModel saved as "
    "'league_match_predictor.pth'"
)


# --------------------------------------------------
# 12. SAMPLE PREDICTION
# --------------------------------------------------

sample = X_test_tensor[0].unsqueeze(0)

model.eval()

with torch.no_grad():

    prediction = model(sample)

probability = prediction.item()

result = (
    "Blue Team Wins"
    if probability >= 0.5
    else "Blue Team Loses"
)

print("\n" + "=" * 50)
print("SAMPLE PREDICTION")
print("=" * 50)

print(f"Win Probability: {probability:.4f}")
print(f"Prediction: {result}")
