from pywsd.similarity import max_similarity
from pywsd.lesk import adapted_lesk
from pywsd.lesk import simple_lesk
from pywsd.lesk import cosine_lesk
from flashtext import KeywordProcessor
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
import requests
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('popular')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('all')


def splitTextToSents(art):
    s=[sent_tokenize(art)]
    s=[y for x in s for y in x]
    s=[sent.strip() for sent in s if len(sent)>15] #Removes all the sentences that have length less than 15 so that we can ensure that our questions have enough length for context
    return s
#print(sents)


def mapSents(impWords,sents):
    processor=KeywordProcessor() #Using keyword processor as our processor for this task
    keySents={}
    for word in impWords:
        keySents[word]=[]
        processor.add_keyword(word) #Adds key word to the processor
    for sent in sents:
        found=processor.extract_keywords(sent) #Extract the keywords in the sentence
        for each in found:
            keySents[each].append(sent) #For each keyword found, map the sentence to the keyword
    for key in keySents.keys():
        temp=keySents[key]
        temp=sorted(temp,key=len,reverse=True) #Sort the sentences according to their decreasing length in order to ensure the quality of question for the MCQ 
        keySents[key]=temp
    return keySents
#print(mappedSents)

def getWordSense(sent,word):
    word=word.lower() 
    if len(word.split())>0: #Splits the word with underscores(_) instead of spaces if there are multiple words
        word=word.replace(" ","_")
    synsets=wn.synsets(word,'n') #Sysnets from Google are invoked
    if synsets:
        wup=max_similarity(sent,word,'wup',pos='n')
        adapted_lesk_output = adapted_lesk(sent, word, pos='n')
        lowest_index=min(synsets.index(wup),synsets.index(adapted_lesk_output))
        return synsets[lowest_index]
    else:
        return None
#print("fin")


def getDistractors(syn,word):
    dists=[]
    word=word.lower()
    actword=word
    if len(word.split())>0: #Splits the word with underscores(_) instead of spaces if there are multiple words
        word.replace(" ","_")
    hypernym = syn.hypernyms() #Gets the hypernyms of the word
    if len(hypernym)==0: #If there are no hypernyms for the current word, we simple return the empty list of distractors
        return dists
    for each in hypernym[0].hyponyms(): #Other wise we find the relevant hyponyms for the hypernyms
        name=each.lemmas()[0].name()
        if(name==actword):
            continue
        name=name.replace("_"," ")
        name=" ".join(w.capitalize() for w in name.split())
        if name is not None and name not in dists: #If the word is not already present in the list and is different from he actial word
            dists.append(name)
    return dists
#print("fin")

def getDistractors2(word):
    word=word.lower()
    actword=word
    if len(word.split())>0: #Splits the word with underscores(_) instead of spaces if there are multiple words
        word=word.replace(" ","_")
    dists=[]
    url= "http://api.conceptnet.io/query?node=/c/en/%s/n&rel=/r/PartOf&start=/c/en/%s&limit=5"%(word,word) #To get ditractors from ConceptNet's API
    obj=requests.get(url).json()
    for edge in obj['edges']:
        link=edge['end']['term']
        url2="http://api.conceptnet.io/query?node=%s&rel=/r/PartOf&end=%s&limit=10"%(link,link)
        obj2=requests.get(url2).json()
        for edge in obj2['edges']:
            word2=edge['start']['label']
            if word2 not in dists and actword.lower() not in word2.lower(): #If the word is not already present in the list and is different from he actial word
                dists.append(word2)
    return dists


file=open("article.txt","r") #"r" deontes read version open
text=file.read()


# tokenize text into individual words and remove stop words
stop_words = set(nltk.corpus.stopwords.words('english'))
tokens = nltk.word_tokenize(text)
tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]

# assign parts of speech to each word
pos_tags = nltk.pos_tag(tokens)

# identify named entities
named_entities = nltk.ne_chunk(pos_tags)

# extract named entities from tree structure
named_entities = [' '.join(leaf[0] for leaf in subtree.leaves())
                for subtree in named_entities
                if hasattr(subtree, 'label') and subtree.label() == 'NE']

# combine named entities with other important words
important_words = named_entities + [word for word, pos in pos_tags if pos.startswith(('N', 'V', 'J'))]
# calculate TF-IDF scores for each word
tfidf = TfidfVectorizer().fit_transform(important_words)
feature_names = TfidfVectorizer().fit(important_words).get_feature_names()
scores = zip(feature_names, tfidf.sum(axis=0).tolist()[0])
sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

# add the top 10 keywords to a list
important_words = []
for word, score in sorted_scores[:10]:
    important_words.append(word)

sents=splitTextToSents(text) #Achieve a well splitted set of sentences from the text article
mappedSents=mapSents(important_words,sents) #Achieve the sentences that contain the keywords and map those sentences to the keywords using this function


mappedDists={}
for each in mappedSents:
    try:
        wordsense=getWordSense(mappedSents[each][0],each) #gets the sense of the word
        if wordsense: #if the wordsense is not null/none
            dists=getDistractors(wordsense,each) #Gets the WordNet distractors
            if len(dists)==0: #If there are no WordNet distractors available for the current word
                dists=getDistractors2(each) #The gets the distractors from the ConceptNet API
            if len(dists)!=0: #If there are indeed distractors from WordNet available, then maps them
                mappedDists[each]=dists
        else: #If there is no wordsense, the directly searches/uses the ConceptNet
            dists=getDistractors2(each)
            if len(dists)>0: #If it gets the Distractors then maps them
                mappedDists[each]=dists
    except:
        pass
#print(mappedDists)

import re
import random

iterator = 1
correct_answers = {}  # store correct answers for each question

# open files for writing
q_file = open("mcq_q.txt", "w")
a_file = open("mcq_a.txt", "w")

print("MCQS")
for each in mappedDists:
    sent = mappedSents[each][0]
    p = re.compile(each, re.IGNORECASE)
    correct_answers[iterator] = each  # store correct answer for the question

    # write question to file
    q_file.write("Question %s:\n" % (iterator))
    op = p.sub("________", sent)
    q_file.write(op + "\n")
    print("Question %s: " % (iterator))
    print(op)

    options = [each.capitalize()] + mappedDists[each]
    options = options[:4]
    random.shuffle(options)

    for i, ch in enumerate(options):
        q_file.write("\t" + chr(ord('A')+i) + ") " + ch + "\n")
        print("\t", chr(ord('A')+i), ") ", ch) 

    user_answer = input("Your answer (A, B, C, or D): ")
    user_answer = user_answer.upper()

    # write answer to file
    a_file.write("%s. %s\n" % (iterator, chr(ord('A') + options.index(each.capitalize()))))

    if user_answer == chr(ord('A') + options.index(each.capitalize())):
        print("Correct!")
    else:
        print("Incorrect. The correct answer is %s.\n" % (chr(ord('A') + options.index(each.capitalize()))))
    q_file.write("\n")
    iterator += 1

print("FILL IN THE BLANKS")
import re

iterator = 1
correct_answers = {}  # store correct answers for each question

# open files for writing
q_file = open("fill_in_the_blank_q.txt", "w")
a_file = open("fill_in_the_blank_a.txt", "w")

print("FILL IN THE BLANK")
for each in mappedDists:
    sent = mappedSents[each][0]
    p = re.compile(each, re.IGNORECASE)
    correct_answers[iterator] = each  # store correct answer for the question

    # write question to file
    q_file.write("Question %s:\n" % (iterator))
    op = p.sub("________", sent)
    q_file.write(op + "\n")
    print("Question %s: " % (iterator))
    print(op)

    user_answer = input("Your answer: ")

    # write answer to file
    a_file.write("%s. %s\n" % (iterator, each))

    if user_answer.lower() == each.lower():
        print("Correct!")
    else:
        print("Incorrect. The correct answer is %s.\n" % each)
    q_file.write("\n")
    iterator += 1

q_file.close()
a_file.close()

# print("TRUE/FALSE QUESTIONS")
# print(mappedDists)
# for each in mappedDists:
#     print(each)
#     sent = mappedSents[each][0]
#     print(sent)
#     # q_file.write("Question %s:\n" % (iterator))
#     # q_file.write(sent + "\n")
#     # print("Question %s: " % (iterator))
#     # print(sent)

#     # user_answer = input("True or False? (T/F): ")
#     # user_answer = user_answer.upper()

#     # # write answer to file
#     # if user_answer == "T":
#     #     a_file.write("%s. True\n" % (iterator))
#     # else:
#     #     a_file.write("%s. False\n" % (iterator))

#     # if user_answer == "T" and each in mappedSents[each][1]:
#     #     print("Correct!")
#     # elif user_answer == "F" and each not in mappedSents[each][1]:
#     #     print("Correct!")
#     # else:
#     #     print("Incorrect. The correct answer is %s.\n" % ("True" if each in mappedSents[each][1] else "False"))
#     # q_file.write("\n")
#     # iterator += 1

# # q_file.close()
# # a_file.close()
