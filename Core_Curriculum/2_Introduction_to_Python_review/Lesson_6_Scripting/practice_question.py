'''
Practice Question:
This will allow you to practice the concepts discussed in the Scripting lesson.

Question: 
Create a function that opens the flowers.txt, 
reads every line in it, and saves it as a dictionary. 
The main (separate) function should take user input 
(user's first name and last name) and parse the user input 
to identify the first letter of the first name. 
It should then use it to print the flower name with the same first letter 
(from dictionary created in the first function).

Sample Output:

>>> Enter your First [space] Last name only: Bill Newman
>>> Unique flower name with the first letter: Bellflower
'''

flowers_file = 'raw_files/flowers.txt'

def file_to_dict(filename):
    flower_dict = dict()
    with open(filename, 'r') as file:
        for line in file:
            flower_info = line.split(':')
            flower_dict[flower_info[0].strip()] = flower_info[1].strip()
    
    return flower_dict

def main():
    name = input('Enter your First [space] Last name only: ')
    key = name.split(' ')[0].strip()[0].upper()
    print(
        f'Unique flower name with the first letter: ' 
        f'{file_to_dict(flowers_file).get(key)}'
    )

if __name__ == '__main__':
    main()
    