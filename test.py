import string

# Define a function to preprocess the text
def preprocess_text(text, to_lower=True):
    sentences = text.split(".")

    # Remove punctuation from each sentence
    for i in range(len(sentences)):
        sentences[i] = sentences[i].translate(str.maketrans("", "", string.punctuation)).replace('\n', '')
    
    # Convert the sentences to lower case
    if to_lower:
        sentences = [s.lower() for s in sentences]
    
    return sentences

# Define a linked list node
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

# Define a linked list class
class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.length = 0
    
    # Add a node to the linked list if it does not already exist
    def add_node(self, value):
        if self.head is None:
            self.head = Node(value)
            self.tail = self.head
            self.length = 1
            return self.head
        else:
            curr_node = self.head
            while curr_node is not None:
                if curr_node.value == value:
                    return curr_node
                if curr_node.next is None:
                    new_node = Node(value)
                    curr_node.next = new_node
                    self.tail = new_node
                    self.length += 1
                    return new_node
                curr_node = curr_node.next
    
    # Get the position of a node in the linked list
    def get_position(self, node):
        curr_node = self.head
        position = 0
        while curr_node is not None:
            if curr_node == node:
                return position
            curr_node = curr_node.next
            position += 1
        return None


file_contents = []
word_list = LinkedList()
word_dict = {}

for i in range(1, 11):
    filename = f"text/{i:02}.txt"
    with open(filename, "r") as f:
        text = f.read()
        text = preprocess_text(text)
        file_contents.append(text)

# Create a linked list of unique words with numerical values
for text in file_contents:
    for sentence in text:
        words = sentence.split()
        for word in words:
            if word not in word_dict:
                word_node = word_list.add_node(word)
                word_dict[word] = word_list.get_position(word_node)

# Create the numerical sequences for each file
file_sequences = []
for text in file_contents:
    file_sequence = []
    for sentence in text:
        sentence_sequence = []
        words = sentence.split()
        for word in words:
            word_sequence = word_dict[word]
            sentence_sequence.append(word_sequence)
        file_sequence.append(sentence_sequence)
    file_sequences.append(file_sequence)

for i, file_seq1 in enumerate(file_sequences):
    for j, file_seq2 in enumerate(file_sequences[i+1:], i+1):
        for x, sentence_seq1 in enumerate(file_seq1):
            for y, sentence_seq2 in enumerate(file_seq2):
                if sentence_seq1 == sentence_seq2:
                    sentence1 = file_contents[i][x].strip()
                    sentence2 = file_contents[j][y].strip()
                    print(f"File {i+1}, sentence {x+1} ({sentence1}) is copied from file {j+1}, sentence {y+1} ({sentence2})")
