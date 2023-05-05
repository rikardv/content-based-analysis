import string
from collections import defaultdict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define a function to preprocess the text
def preprocess_text(text, to_lower=True):
    sentences = text.split(".")

    # Remove punctuation from each sentence
    for i in range(len(sentences)):
        sentences[i] = sentences[i].translate(str.maketrans("", "", string.punctuation)).replace('\n', ' ')
    
    # Convert the sentences to lower case
    if to_lower:
        sentences = [s.lower() for s in sentences]
    
    return sentences


file_contents = []

for i in range(1, 11):
    filename = f"text/{i:02}.txt"
    with open(filename, "r") as f:
        text = f.read()
        text = preprocess_text(text)
        file_contents.append(text)

# print(file_contents[0][168])
# print(file_contents[6][72])

# Create a word list with numerical values
word_list = defaultdict(lambda: len(word_list))
sentence_list = {}

for text in file_contents:
    for sentence in text:
        words = sentence.split()
        for word in words:
            word_list[word]

doc = 0
sent = 0
doc_arr = []
sent_arr = []
for text in file_contents:
    for sentence in text:
        if not sentence in sentence_list:
            doc_arr.append(doc)
            sent_arr.append(sent)
            sentence_list[sentence] = {"document": doc_arr,"sentence":sent_arr}
            doc_arr = []
            sent_arr = []
        else:
            if doc not in sentence_list[sentence]["document"]:
                sentence_list[sentence]["document"].append(doc)
                sentence_list[sentence]["sentence"].append(sent)
        
        sent = sent+1
    doc = doc +1  
    sent = 0

    



#create dataset
inputDoc = input("which document to you want to investigate?")
test =np.asarray(file_contents[int(inputDoc)][:])
test2 = np.array([])
for key in test:
    
    if len(key) <20:
        test2 = np.append(test2,1)
    else:
        test2 = np.append(test2,len(sentence_list[key]["document"]))
# print(test2)
test =test2
columns = 10
for i in range(columns-len(test)%columns):
    test = np.append(test,0)
test = test.reshape(-1, columns)

# print(test2.shape)
# Create a dataset
df = pd.DataFrame(test)

# print(df)
#print sentence information
for key in sentence_list:
    if len(sentence_list[key]["document"]) > 1 and len(key) > 20:
        print("This senctence was found  : \n" + key)
        print("in the following documents: ")
        print(sentence_list[key]["document"])
        print('\n')

p1 =sns.heatmap(df, linewidths=2, square=True, cmap='jet')

plt.show()

