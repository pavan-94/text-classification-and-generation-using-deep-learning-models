# text-classification-and-generation-using-deep-learning-models
Introduction:
This report is subjected to various models of deep learning for sentiment analysis and language generation. For this purpose, the dataset used is movie reviews from IMDB which is obtained from https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews. The dataset consists of two columns namely ‘reviews’ and ‘sentiment’ and a total of 50,000 rows. Google colab is used for coding as the free GPU provided by Google is very effective and helps in fast training of the models. The dataset is downloaded and uploaded to google drive in order to use the data in colab. 

Problem Statement: Classification of IMDB review into two classes namely ‘positive’ and ‘negative’ using various text classification models.
Problem Statement: Using LSTM and n-grams to predict next character in order to generate sentences.
Two separate notebooks are created one with Part 1 and first section of Part 2 and second notebook consists of last section of Part2

Part 1: IMDB Modelling Task:
The first task deals with building of various text classification models and analyzing their performance in order to determine the best performing model for text classification of the IMDB reviews into positive and negative. In order to use any file from google drive onto colab we must mount the google drive by giving the path as on where to mount. Once the drive is mounted dataset in CSV format is read to a data frame using panads read_csv. Following models are built:
Single Layer LSTM
Multi-Layer LSTM
Simple RNN
Multi-channel CNN with Kernel Size 2 and 5
Multi-channel CNN+LSTM model
Embeddings Learnt on Fly
Gnews swivel sentence embeddings from TensorFlow hub 

Part 2: Working with your own Data 
In this part of the project a new dataset is created manually by collecting 60 reviews from 30 movies. One review each for positive and negative is collected from each movie such that the dataset is balanced with 30 positive and 30 negatives. The movies selected are from the year 1994. 
Since the best performed model in previous section was with pre-trained embeddings, in this section the new dataset is used to do sentiment analysis from the saved model and building the model from scratch. Initially the dataset is loaded from the google drive and the sentiments are encoded as 0 and 1 representing positive and negative.  No pre-processing of the text was done as pre-trained embedding can pre-process the data and creating sentence embedding from the raw text. The dataset is split into 70% training and 30% validation set. 

Part 3: Writing your own Reviews
In this part of the project LSTM and n-gram model as the statistical model choice is used to generate some text using the original IMDB dataset of 50,000 reviews. The original review text is split into 3 parts one set having only positive reviews, one with only negative reviews and one containing all the reviews. LSTM and n-grams models are created for each of the data set which brings up a total of 6 models. The data is not split into test or validation sets. 

References 
Recurrent Layers - Keras Documentation. (2019). Retrieved 1 May 2020, from https://keras.io/layers/recurrent/ 

Brownlee, J. (2019a). Dropout Regularization in Deep Learning Models With Keras. Dropout Regularization in Deep Learning Models With Keras. https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/ 

Brownlee, J. (2019b). Stacked Long Short-Term Memory Networks. Stacked Long Short-Term Memory Networks. https://machinelearningmastery.com/stacked-long-short-term-memory-networks/ 

Eckhardt, K. (2018). Choosing the right Hyperparameters for a simple LSTM using Keras. Choosing the Right Hyperparameters for a Simple LSTM Using Keras. https://towardsdatascience.com/choosing-the-right-hyperparameters-for-a-simple-lstm-using-keras-f8e9ed76f046 

Language Modeling. (2019). Language Modeling. https://github.com/chakki-works/chariot/blob/master/notebooks/language%20modeling.ipynb 

Rosebrock, A. (2019). Why is my validation loss lower than my training loss? https://www.pyimagesearch.com/2019/10/14/why-is-my-validation-loss-lower-than-my-training-loss/ 

Text generation with an RNN. (2020). Text Generation with an RNN. https://www.tensorflow.org/tutorials/text/text_generation 

Tan, L. (n.d.). N-gram Language Model with NLTK. N-Gram Language Model with NLTK. https://www.kaggle.com/alvations/n-gram-language-model-with-nltk 

Text classification with TensorFlow Hub: Movie reviews. (2020). Text Classification with TensorFlow Hub: Movie Reviews. https://www.tensorflow.org/tutorials/keras/text_classification_with_hub 

Verman, A. (n.d.). Creating custom corpus in NLTK using CSV file. Creating Custom Corpus in NLTK Using CSV File. https://stackoverflow.com/questions/30357899/creating-custom-corpus-in-nltk-using-csv-file 


