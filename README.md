# Multi-Media Questionnaire Generator

The multimedia quiz generator project is a software application that can generate multiple choice questions (MCQs), fill-in-the-blank questions, and true or false questions from various sources such as text phrases, PDF documents, videos, and audio files. The project uses natural language processing algorithms to extract relevant information from the source and generate questions based on that information. The system can also use Wikipedia to gather additional information on the topic. The generated questions are presented to the user in a quiz format and can be modified or added to by the user. The project is a useful tool for testing knowledge and can be used in educational settings, online learning platforms, or as a fun activity.

## QuestionnaireGenerator.ipynb

The file "QuestionnaireGenerator.ipynb" is a Jupyter Notebook that contains different sections for generating questions from text using natural language processing techniques. The headings in the notebook are as follows:

- **Question Generation Model**: This section explains how to create a question generation model. It includes the following sub-sections:
    
    - **Preparing dataset**: The SQUAD dataset can be downloaded and preprocessed to extract the passages and their corresponding questions and answers. The extracted passages can be used as input to the T5 model.

    - **Preparing dataset to load into model**: The prepared dataset is loaded into the model for training.

    - **Fine-tune the T5 model**: The T5 model can be fine-tuned on the SQUAD dataset using a question generation objective. During fine-tuning, the model learns to generate questions based on the given passage.

    - **Saving the model**: The trained model is saved for future use.

    - **Testing the model**: This sub-section demonstrates how to test the performance of the trained model on new data.

    - **Evaluating the model**: In this sub-section, the accuracy and other metrics of the trained model are evaluated.

- **Text Generator**: The text generator can use the Wikipedia API to extract text from Wikipedia pages based on a given search term or topic. It can also use the Whisper API to extract text from audio and video files by transcribing the spoken words into text. Additionally, the text generator can extract text from PDFs.

- **Text Summarizer**: This section explains how to summarize text using the TextRank algorithm. It demonstrates how to extract the most important sentences from a given text and use them to generate a summary.

- **Extraction of keywords**: This section explains how to extract important keywords from a given text. It demonstrates how to use the RAKE library to extract keywords and their frequency from a text.

- **Sentence mapping**: This section explains how to map sentences in a given text to generate questions. It demonstrates how to use the flashtext library to identify sentence structures and extract relevant information for generating questions.

- **Question Generation:** This section explains how to generate different types of questions using the previously generated models. It includes the following sub-sections:

    - **MCQ Generation**: This sub-section explains how to generate multiple-choice questions using the question generation model.

    - **Fill in the blanks**: Here, the fill-in-the-blank questions are generated using the extracted keywords and sentence mapping techniques.

    - **True or False**: This sub-section explains how to generate true or false questions based on the given text and the identified sentence structures.

## DistractorGeneration.ipynb

DistractorGeneration.ipynb is a Jupyter Notebook that contains a Python code for generating distractors for multiple-choice questions. The notebook is designed to work in conjunction with the QuestionnaireGenerator.ipynb notebook to generate complete multiple-choice questions with distractors.

The script uses various natural language processing (NLP) techniques to generate distractors for a given list of answers. It leverages pre-trained models, such as sense2vec and Gensim's Doc2Vec, to extract relevant information and generate distractors.

The first part of the code installs required packages, including sense2vec, wn, transformers, spacytextblob, and nltk. It then imports various libraries, such as Tensorflow, numpy, and Tokenizer, and defines some functions, such as WordNetLemmatizer and PorterStemmer, for text preprocessing.

The script then reads in two text files, "QandA.txt" and "keywords.txt", containing a list of questions and answers and a list of keywords, respectively. It preprocesses the data, removes duplicates, and creates a dictionary to store distractors for each answer.

The script then uses the sense2vec model to generate distractors for each answer. It first checks if the answer is in the sense2vec vocabulary. If not, it generates distractors by finding the most similar words for each word in the answer. It uses lemmatization and stemming to ensure that the generated distractors are not exact matches of the answer.

The script then uses Gensim's Doc2Vec model to generate additional distractors. It first downloads the "text8" dataset and trains the model on this data. It then generates a list of distractors by finding the most similar words to the target word using cosine similarity.

Finally, the script saves the generated distractors to a text file.

## Running the Code

- Go to [Google Colab](https://colab.research.google.com/).
- Click on "File" and then "Upload notebook".
- Connect to a runtime

For QuestionnaireGenerator.ipynb
- Upload files present in [/files](./files/)
- Run 

For DistractorGeneration.ipynb
- Upload files: QandA.txt, keywords.txt and context.txt
- Run
