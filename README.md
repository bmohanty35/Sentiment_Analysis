# Sentiment Analysis using Logistic Regression (Completed Steps)
## Step 1: Installed Required Packages
I installed the required Python packages including:

nltk for natural language processing

seaborn and matplotlib for visualizations

scikit-learn for model building and evaluation

## Step 2: Imported Necessary Libraries
Imported all essential libraries such as:

pandas, numpy for data manipulation

re, nltk.corpus.stopwords for text processing

TfidfVectorizer, train_test_split, LogisticRegression from sklearn for machine learning

accuracy_score, classification_report for evaluation

## Step 3: Downloaded NLTK Stopwords
Downloaded the English stopwords corpus from NLTK to use during text preprocessing.

## Step 4: Loaded and Cleaned the Dataset
Loaded the Twitter sentiment dataset from a public GitHub link.

Selected only the label and tweet columns

Renamed the columns to label and text

Removed any rows with null tweet content

## Step 5: Balanced the Dataset
Identified an imbalance in class distribution and balanced it by undersampling the majority class so both classes (Positive = 0, Negative = 1) had equal representation.

## Step 6: Preprocessed the Tweets
Defined and applied a text cleaning function that:

Removed URLs, mentions, hashtags

Removed non-alphabetic characters

Converted text to lowercase

Removed English stopwords

Stored the result in a new column clean_text.

## Step 7: Converted Text to TF-IDF Vectors
Used TfidfVectorizer to convert the cleaned tweets into numerical feature vectors.

Configured to use unigrams and bigrams

Limited to the top 5000 most frequent features

## Step 8: Split the Dataset
Split the TF-IDF features and labels into training and testing sets using an 80-20 ratio with a fixed random seed (random_state=42).

## Step 9: Trained Logistic Regression Model
Trained a logistic regression classifier on the training data with a maximum of 1000 iterations to ensure convergence.

## Step 10: Evaluated the Model
Tested the trained model on the unseen test set.

Achieved an accuracy of ~85.7%

Generated a detailed classification report showing high precision and recall for both sentiment classes

## Step 11: Defined Sentiment Prediction Function
Created a predict_sentiment() function that:

Accepts raw input text

Cleans and vectorizes the input

Predicts sentiment using the trained model

Returns "Positive" if the predicted label is 0, and "Negative" if 1

## Step 12: Performed Sample Predictions
Tested the function on several example texts.
'Amazing performance by the actor.' => Positive  
'I hate this so much.' => Negative  
'Wonderful service, thank you!' => Positive  
