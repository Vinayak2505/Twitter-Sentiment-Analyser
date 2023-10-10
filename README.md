# Twitter-Sentiment-Analyser
Twitter is a social media platform where users share their opinions and thoughts on various topics. With the enormous amount of data generated every minute on Twitter, sentiment analysis has become an essential tool for businesses and organizations to understand their audience's opinions and emotions about their products or services. This project aims to classify the polarity of tweets into positive, negative, or neutral. This project consists of 3 phases:

Tweet Extraction:
We use Selenium to mimic user interactions and extract data from the Twitter platform in an automated manner. It is an open-source automation testing software which used for Web Scraping, Software Testing, Web Development, etc. It allows developers to control web browsers programmatically, interact with web elements, navigate through web pages, and extract information from websites.

It begins by setting up the WebDriver with the ChromeDriver executable and creating an instance of WebDriver using Chrome. The script then proceeds to log into a Twitter account by providing the username and password. If an unusual activity page appears, it handles it accordingly. After successful login, the script performs a search for a specified user and navigates to their profile. It defines two functions: remove_tags and rt, which are used to translate and clean the tweet text by removing HTML tags, URLs, and non-alphanumeric characters.

Next, the script enters a loop to gather tweets from the user's account. It retrieves the tweet text, cleans it using the defined functions, and appends it to a variable. The loop continues until it collects the desired number of tweets. To ensure all tweets are retrieved, the script scrolls the page and waits for more tweets to load before extracting them. Once the loop completes, the WebDriver instance is closed.

Model Training and Testing:
The machine learning model uses the Naive Bayes algorithm. It is a supervised machine learning based on Bayes' theorem which assumes that each feature (i.e., word) in the input data is conditionally independent of every other feature, given the class label (i.e., sentiment). 

1.	Data Collection: We load a dataset containing a number of tweets and their respective sentiments. This dataset will be used to train our ML Model.
2.	Data Balancing: We perform a visualization to check if the data is evenly distributed data or not. Balanced data is highly important for training a model accurately. To balance the dataset, the oversample() function is defined. It oversamples the minority classes (neutral and negative tweets) by sampling with replacement to match the number of positive tweets.
3.	Data Cleaning: The various processes required to clean the tweets is done using NLTK. Natural Language Toolkit is a popular open-source Python library for text processing, part-of-speech tagging, parsing, and sentiment analysis. The tweets are cleaned by removing HTML tags, URLs, and non-alphanumeric characters. 

  a.	Stop-word removal: - Stop-words are commonly used words that do not carry much meaning. They are removed from the tweets using the NLTK library's stop-words corpus. E.g. - "the", "and", "a", "an", etc. This improves upon accuracy of the model. 

  b.	Stemming: - It is the process of reducing words to their base or root form, called the stem. The stem may not necessarily be a valid word on its own but represents the core meaning of the word. Stemming typically involves removing suffixes and prefixes from words to obtain the stem.
  
  c.	Lemmatization: - It is the process of reducing words to their base or dictionary form, known as the lemma. Unlike stemming, lemmatization ensures that the resulting lemma is a valid word, retaining its meaning. This involves considering the word's part of speech (POS) and applying morphological analysis.
5.	Data Exploration: We explore and visualize the dataset to get a little understanding of the data provided. We create a plot of the top 50 words for each sentiment class. This gives us an idea about the data we have collected. 
6.	Selection of Hyperparameter Values: We find the best 'k' value for feature selection and the best 'alpha' value for smoothing in the Naive Bayes model.       For selecting the 'k' value, the code iterates through a list of 'k' values and evaluates the performance of the model using each 'k' value. The accuracy of the model is calculated for each 'k' value, and a bar chart is created to visualize the accuracy for each 'k' value. The 'k' value with the highest accuracy is considered the best 'k' value. The best ‘alpha’ value is calculated in a similar manner. The final pipeline is defined using the best 'k' value and 'alpha' value.
7.	Model Definition: In this case, the pipeline consists of three main steps: vectorization, feature selection, and classification.
  
  a.	Vectorization: This step uses the TF-IDF Vectorizer class from scikit-learn which is a text feature extraction method that converts text documents into numerical feature vectors. It converts a collection of raw documents into a matrix of TF-IDF (Term Frequency-Inverse Document Frequency) features. TF-IDF represents the importance of a term in a document within a collection of documents.
  
  b.	Feature Selection: The second step is the "selector" step, which uses the SelectKBest class which is a feature selection method that selects the top k features based on a scoring function. In this case, the scoring function used is chi2, which is the chi-square test statistic. The chi2 test measures the independence between categorical variables. The SelectKBest with chi2 selects the k features with the highest chi-square scores.
  
  c.	Classification: The final step in the pipeline uses the MultinomialNB class to predict the sentiments of the cleaned pre-processed and feature-extracted tweets.
8.	Training and Testing: We follow an incremental approach to train our model. This technique involves training the model on a small subset of the data and then gradually increasing the size of the training data as the model becomes more accurate. At each step, the code trains and tests the model on the current dataset, calculates accuracy, precision, recall, F1 score, and displays the results in a table and confusion matrix.
  
  a.	Precision: - It is the ratio of true positives to the total number of predicted positive instances. It measures the accuracy of positive predictions.
  
  b.	Recall: - It is the ratio of true positives to the total number of actual positive instances. It measures the ability of the classifier to find all positive instances.
  
  c.	F1-score: - It is the harmonic mean of precision and recall. F1 score ranges between 0 and 1, where 1 is the best possible score.
  
  d.	Confusion Matrix: - It shows the number of correct and incorrect predictions made by the model compared to the true outcomes.
 
Sentiment Prediction:
We apply the trained pipeline to the collected cleaned and translated tweets. The model processes the input and predicts sentiment category for the tweet. The sentiment categories can be positive, negative, or neutral. These results are displayed as tweets and its predicted sentiment and is visualised with the help of a bar graph which is drawn using matplotlib. The bar graph indicates the number of positive, negative and neutral tweets. We then store these tweets and their sentiments in a csv file.

