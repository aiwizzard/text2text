from collections import Counter
import json
import os

data_dir = '.data/'

corpus_movie_conv = os.path.join(os.getcwd(), data_dir, 'cornell movie-dialogs corpus/movie_conversations.txt')
corpus_movie_lines = os.path.join(os.getcwd(), data_dir, 'cornell movie-dialogs corpus/movie_lines.txt')
output_path = os.path.join(os.getcwd(), data_dir, 'train_data.json')
wordmap_out = os.path.join(os.getcwd(), data_dir, 'wordmap.json')

max_len = 25


with open(corpus_movie_conv, 'r', encoding='iso-8859-1') as c:
    conv = c.readlines()

with open(corpus_movie_lines, 'r', encoding='iso-8859-1') as l:
    lines = l.readlines()



lines_dic = {}
for line in lines:
    objects = line.split(" +++$+++ ")
    lines_dic[objects[0]] = objects[-1]


def remove_punc(string):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = ""
    for char in string:
        if char not in punctuations:
            no_punct = no_punct + char  # space is also a character
    return no_punct.lower()


pairs = []
for con in conv:
    ids = eval(con.split(" +++$+++ ")[-1])
    for i in range(len(ids)):
        qa_pairs = []
        
        if i==len(ids)-1:
            break
        first = remove_punc(lines_dic[ids[i]].strip())
        second = remove_punc(lines_dic[ids[i+1]].strip())
        qa_pairs.append(first.split()[:max_len])
        qa_pairs.append(second.split()[:max_len])
        pairs.append(qa_pairs)

word_freq = Counter()
for pair in pairs:
    word_freq.update(pair[0])
    word_freq.update(pair[1])

min_word_freq = 3
words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
word_map = {k: v + 1 for v, k in enumerate(words)}
word_map['<unk>'] = len(word_map) + 1
word_map['<start>'] = len(word_map) + 1
word_map['<end>'] = len(word_map) + 1
word_map['<pad>'] = 0

with open(wordmap_out, 'w') as file:
    json.dump(word_map, file)

def encode_question(words, word_map):
    enc_c = [word_map.get(word, word_map['<unk>']) for word in words] + [word_map['<pad>']] * (max_len - len(words))

def encode_reply(words, word_map):
    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in words] + [word_map['<end>']] + [word_map['<pad>']] * (max_len - len(words))
    return enc_c

encoded_pairs = []
for pair in pairs:
    first = encode_question(pair[0], word_map)
    second = encode_reply(pair[1], word_map)
    encoded_pairs.append([first, second])

with open(output_path, 'w') as file:
    json.dump(encoded_pairs, file)
