# Spam Detection using LSTM with PyTorch
This project implements a spam detection model using a Long Short-Term Memory (LSTM) network built with PyTorch. The dataset consists of email or message text and corresponding labels indicating whether the message is spam or not. We preprocess the text, convert it into numerical vectors, and train an LSTM model to classify the messages. The final model is saved for future use to predict whether a given message is spam.

## Requirements
#### Before running the project, ensure you have installed the following dependencies:

pip install torch torchvision torchaudio \
pip install scikit-learn \
pip install tqdm \
pip install nltk 

## Data
#### The dataset used is combined_data.csv, which contains two columns:
- text: The message content.
- label: The target column with binary values (0 for not spam, 1 for spam).

#### Combined_data.csv is downloaded from Kaggle.com

## Preprocessing

1. Text Cleaning and Tokenization: The raw text is cleaned, tokenized, and stopwords are removed. Stemming is applied using the NLTK PorterStemmer. 
2. Vectorization: The CountVectorizer from scikit-learn is used to convert the preprocessed text into numerical vectors with a maximum of 5000 features. 
3. Label Encoding: The labels are encoded to 0s and 1s using LabelEncoder. 
4. Train-Test Split: The data is split into training (80%) and test (20%) sets. 

## Model Architecture
#### The model is built using PyTorch, with the following architecture:

1. LSTM Layer: The input is passed through an LSTM with 128 hidden units. 
2. Fully Connected Layer: The LSTM output is passed through a fully connected layer. 
3. Sigmoid Activation: The final output is passed through a sigmoid function to predict the probability of a message being spam. 

## Training
- The model is trained using the Binary Cross-Entropy Loss (BCELoss).
- The optimizer used is Adam.
- Training runs for 10 epochs, with a batch size of 64.

## Evaluation
#### After training, the model is evaluated on the test set, and the following metrics are calculated:
- Accuracy
- Precision
- Recall
- F1 Score
##### The evaluation function evaluate_model() computes these metrics.

### Conclusion
This project demonstrates the use of an LSTM model for text classification, specifically for spam detection. The model achieves high accuracy on the test data and can be easily adapted to detect spam in real-world applications.





