'''
Write a function called generate_password that selects three random words 
from the list of words word_list and concatenates them into a single string. 
Your function should not accept any arguments and should reference the 
global variable word_list to build the password.
'''

import random

word_file = 'raw_files/words.txt'
word_list = []

# filling up word_list from word_file
with open(word_file, 'r') as words:
    for line in words:
        word = line.strip().lower()
        if 3 < len(word) < 8:
            word_list.append(word)

def generate_password():
    return ''.join(random.sample(word_list, 3))

# print(random.choices(word_list, k=3))
print(generate_password())