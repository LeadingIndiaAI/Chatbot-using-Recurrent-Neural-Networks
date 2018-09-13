# ChatCrazie


## Overview
This is our ChatBot: ChatCraZie on Youtube. It starts with [this](https://youtu.be/0TGp_CrSSxg) video on Youtube.<br>

A chatbot implemented using RNN and GloVe embeddings whch answers your query crazily
<br>
To download Glove Embeddings, go to this [LINK](http://nlp.stanford.edu/data/glove.6B.zip) <br>


=======
# PROBLEM STATEMENT 

Main problem domain is building a Chatbot, which is capable of generating the best response for any general user query. The “best” reply must contain following attributes:

1.Answers to the user’s question.<br>
2.Provides sender with relevant details.<br>
3.Able to ask follow-up questions, and <br>
4.Able to continue the conversation in a realistic manner. <br>In order to achieve this goal, The Chatbot needs to have understanding of the sender’s messages so that it can predict which sort of response will be relevant and it must be correct lexically and grammatically while generating the reply.<br>

## -METHODOLOGY
#### ➢ DATA SET ACQUISITION: 
It is the one of the most important step in designing a chatbot. After lots of research and fine tuning experimentation, we have used gunthercox dataset and also modified it. The dataset contains conversations based on various topics such as emotions, psychology, sports, normal day-to-day conversations etc. The better the dataset, the more accurate and efficient conversational results can be obtained.

#### ➢ PRE-PROCESSING:
This step involves cleaning the dataset such as removing unwanted characters (- or – or # or $ etc) and replacing them with blank spaces.

#### ➢ TOKENIZATION AND VECTORIZATION:
Basic tokenizer (such as in nltk) splits the text into minimal meaningful units called as tokens, such as words and eliminates punctuation characters. We have used GloVe (Global Vectors for word representations), which is an unsupervised learning algorithm that maps words to vectors of real numbers.

#### ➢ SPLITTING TRAIN AND TEST DATA:
We divided the total dataset into 80% training data and 20% validation data. The chatbot was tested based on comparative study of review analysis from various users, as how relevant they find the conversation to be with the chatbot.

#### ➢ CREATION OF LSTM, ENCODER AND DECODER MODEL:
LSTM are a special kind of RNN which are capable of learning long-term dependencies. Encoder-Decoder model contains two parts- encoder which takes the vector representation of input sequence and maps it to an encoded representation of the input. This is then used by decoder to generate output.

#### ➢ TRAIN AND SAVE MODEL: 
We trained the model on modified gunthercox dataset, with 250 epochs and batch size of 16, word embedding size was set to100, we took categorical crossentropy as our loss function and optimiser used was rmsprop. We got
the best results with these parameters. We trained and tested our model on NVIDIA DGX-1 V100. Training accuracy obtained was approximately 99% and validation accuracy of about 97%.

#### ➢ PREDICTION:
Finally the user can input one’s questions and converse with the chatbot. The results obtained are satisfactory according to review analysis.


This project was done under the Guidance of [Mr. Shreyans Jain](https://github.com/shreyanse081).<br> 
### Submitted by: [Kushagra Goel](https://github.com/kushagra2101), [Anushree Jain](https://github.com/anushreejain98), [Akshita Gupta](https://github.com/akshitagupta114).
=======

