# -League-of-Legends-Match-Predictor
A deep learning project that predicts the outcome of League of Legends matches using team statistics and a PyTorch neural network.
# 🎮 League of Legends Match Predictor

A Deep Learning project that predicts the outcome of a League of Legends match using in-game statistics and a PyTorch neural network.

## 📌 Project Overview

This project uses machine learning and deep learning techniques to predict whether the **Blue Team will win or lose** a League of Legends match.

The model is trained using match statistics collected during the first 10 minutes of a game, such as:

* Gold difference
* Experience difference
* Number of kills
* Number of deaths
* Towers destroyed
* Dragons
* Heralds
* Wards placed and destroyed
* Other team performance statistics

## 🎯 Objective

The objective is to predict the target variable:

```text
blueWins
```

Possible values:

* `1` → Blue Team Wins
* `0` → Blue Team Loses

## 🧠 Model Architecture

The project uses a PyTorch neural network with:

* Input Layer
* Dense Layer with 64 neurons
* ReLU Activation
* Dropout Layer
* Dense Layer with 32 neurons
* ReLU Activation
* Dropout Layer
* Dense Layer with 16 neurons
* Output Layer with Sigmoid Activation

The Sigmoid activation produces a probability between 0 and 1 representing the probability of the Blue Team winning.

## 🛠️ Technologies Used

* Python
* PyTorch
* Pandas
* Scikit-learn

## 📂 Project Structure

```text
league-of-legends-match-predictor/
│
├── high_diamond_ranked_10min.csv
├── league_match_predictor.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/league-of-legends-match-predictor.git
```

Move into the project directory:

```bash
cd league-of-legends-match-predictor
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 📊 Dataset

The project uses the **League of Legends Diamond Ranked Games dataset**.

The dataset should be saved in the project directory as:

```text
high_diamond_ranked_10min.csv
```

The data contains match statistics from the first 10 minutes of ranked League of Legends games.

## 🚀 Running the Project

Run:

```bash
python league_match_predictor.py
```

The program will:

1. Load the dataset.
2. Clean and preprocess the data.
3. Split the dataset into training and testing sets.
4. Standardize the numerical features.
5. Build a PyTorch neural network.
6. Train the model.
7. Evaluate the model.
8. Display a sample match prediction.
9. Save the trained model.

## 📈 Evaluation

The model is evaluated using:

* Binary Cross-Entropy Loss
* Test Accuracy

A prediction probability greater than or equal to `0.5` is classified as:

```text
Blue Team Wins
```

Otherwise:

```text
Blue Team Loses
```

## 🚀 Future Improvements

Possible improvements include:

* Hyperparameter tuning.
* Adding batch training using DataLoader.
* Using additional neural network architectures.
* Comparing results with Random Forest and XGBoost.
* Creating a web interface for match predictions.
* Using real-time League of Legends match data.

## 👨‍💻 Author

**Yash Vardhan**

B.Tech CSE (AI)
VIT Bhopal University

## 📜 License

This project is created for educational and learning purposes.
