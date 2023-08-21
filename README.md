# Natural Language Processing (Data Analysis on Text Data)
Performing data analysis on text data.

## 1. Tokenization
Tokenization is the procedure of splitting text into a set of meaningful fragments. These pieces are called tokens.
### Output:
![image](https://user-images.githubusercontent.com/116111985/196520793-12d8012f-ade0-4b2e-8146-76cd66c95eb0.png)

## 2. Stop words Removal
Stop words are the words which are commonly filtered out before processing a natural language. These are the most common words in any language (like articles, prepositions, pronouns, conjunctions, etc) and does not add much information to the text. Examples of a few stop words in English are “the”, “a”, “an”, “so”, “what”.

Removal of stop words certainly reduces the dataset size and gives more emphasis to the crucial information for detailed analysis.
### Output:
![image](https://user-images.githubusercontent.com/116111985/196521052-d275c2ef-3d4f-4e12-b8a5-6f01679d1ea0.png)

## 3. Stemming
Stemming is one of the most common data pre-processing operations we do in text preprocessing. Stemming is the process of removing a part of a word or reducing a word to its stem or root. We use a few algorithms to decide how to chop a word off.
### Output:
![image](https://user-images.githubusercontent.com/116111985/196521302-f6ff2c25-4fef-45b2-8cae-b8d1a6146044.png)

## 4. Lemmatization
Lemmatization is like stemming in reducing inflected words to their word stem but differs in the way that it makes sure the root word (also called as lemma) belongs to the language.

As a result, this one is generally slower than stemming process. Since lemmatization requires the part of speech, it is a less efficient approach than stemming.
### Output:
![image](https://user-images.githubusercontent.com/116111985/196521850-351bf8ad-6b2f-4972-b3ff-f0b9077fe3b1.png)

## 5. TF-IDF (Word Cloud)
TF-IDF is a statistical measure that evaluates how related a word is to a document in a set of documents.
### Output:
![image](https://user-images.githubusercontent.com/116111985/196522099-03eea90e-59f8-48b1-a366-01c1949192a4.png)

## 6. N-Grams
An n-gram is a contiguous sequence of n items in the text. In our case, we will be dealing with words being the item, but depending on the use case, it could be even letters, syllables, or sometimes in the case of speech, phonemes.
### Output:
![Screenshot from 2022-10-12 10-55-30](https://user-images.githubusercontent.com/116111985/196522463-ab2fe7fe-45bd-4809-a4c4-05d62e1fe55f.png)

## 7. VSM (Vector Space Model)
VSM is a statistical model for representing text information for Information Retrieval, NLP, Text Mining.

Representing documents in VSM is called "vectorizing text" contains the following information: how many documents contain a term, and what are important terms each document has.
### Output:
![image](https://user-images.githubusercontent.com/116111985/196522884-f9484ce6-5e94-4b5a-aecb-2ee29b06a34b.png)

# Dataset Source
Author: Larxel

URL: www.kaggle.com/datasets/andrewmvd/udemy-courses

## Dataset Description:
This dataset contains records of courses from 4 subjects (Business Finance, Graphic Design, Musical Instruments and Web Development) taken from Udemy. 
Udemy is a massive online open course (MOOC) platform that offers both free and paid courses. Anybody can create a course, a business model by which allowed Udemy to have hundreds of thousands of courses.
This version modifies column names, removes empty columns, and aggregates everything into a single csv file for ease of use.
