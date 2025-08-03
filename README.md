# flipkart_review_sentiment_analysis

This project performs sentiment analysis on Flipkart product reviews using a machine learning approach. The goal is to classify reviews as either positive or negative based on their content and associated ratings.


## Introduction

Customer reviews are invaluable for understanding product perception and driving business decisions. This project automates the process of gauging customer sentiment from textual reviews. By classifying reviews as positive or negative, businesses can quickly identify areas of strength and weakness in their products.

## Dataset

The dataset used for this analysis is `flipkart_data.csv`. It contains two main columns:

- `review`: The text content of the customer review.
- `rating`: The numerical rating given by the customer (typically on a scale of 1 to 5).

The sentiment labels are derived from the `rating` column:
- **Positive Sentiment (1):** Reviews with a rating of 4 or 5.
- **Negative Sentiment (0):** Reviews with a rating of 1, 2, or 3.



## Methodology

The sentiment analysis is performed using the following steps, as outlined in the Jupyter Notebook:

1.  **Data Loading**: The `flipkart_data.csv` file is loaded into a pandas DataFrame.
2.  **Preprocessing**:
    * Review text is converted to lowercase.
    * Stop words (common words like "the", "a", "is", etc.) are removed to focus on more meaningful terms. 
    * A `sentiment` column is created based on the `rating` column, where ratings $\ge 4$ are considered positive (1) and ratings $< 4$ are considered negative (0).
3.  **Exploratory Data Analysis (EDA)**:
    * A bar plot visualizes the distribution of sentiments (positive vs. negative) in the dataset.
    * A word cloud is generated for positive reviews to highlight frequently occurring words, providing insights into common positive feedback.
4.  **Feature Extraction**:
    * `TfidfVectorizer` is used to convert the preprocessed text reviews into numerical features. This technique reflects the importance of a word in a document relative to the entire corpus. `max_features` is set to 5000.
5.  **Model Training**:
    * The dataset is split into training and testing sets (80% training, 20% testing). 
    * A `DecisionTreeClassifier` model is trained on the training data.
6.  **Model Evaluation**:
    * The trained model's performance is evaluated on the test set. 
    * `accuracy_score` is calculated to measure the overall correctness of the model's predictions.
    * A confusion matrix is generated and visualized using a heatmap to show the number of correct and incorrect predictions for each class (positive/negative).

## Results

The sentiment analysis model achieved an accuracy of approximately **85.47%**. 

The confusion matrix provides a detailed breakdown of the model's performance:

The word cloud for positive reviews shows prominent words such as "sound," "bass," "quality," "good," and "product," indicating these are key aspects of highly-rated items.

# Dependencies

The project relies on the following Python libraries:

-   `pandas`
-   `nltk`
-   `scikit-learn`
-   `matplotlib`
-   `seaborn`
-   `wordcloud`
