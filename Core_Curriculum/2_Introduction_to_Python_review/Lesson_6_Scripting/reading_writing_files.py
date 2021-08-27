# example 1 reading file
file1 = open('raw_files/some_file.txt', 'r')
file_data = file1.read()
file1.close()
print(file_data)

# example 2 open multiple files and close them
# files = []
# for i in range(1, 10000):
#     file2 = open('some_file.txt', 'r')
#     files.append(file)
#     file2.close() # we should close any file after we done with it,
#     # otherwise it will take resources and rais an error of too many files
#     print(i)

print('-' * 40)
# example 3 write to a file
file3 = open('raw_files/another_file.txt', 'w')
file3.write('Hello, there!')
file3.close()

# example 4 open file with context manager
with open('raw_files/another_file.txt', 'r') as file4:
    file_data4 = file4.read()
print(file_data4)

print('-' * 40)
# example 5 reading a chanck of string (chars) from a file using read method
with open('raw_files/camelot.txt', 'r') as song:
    print(song.read(2))
    print(song.read(8))
    print(song.read())

print('-' * 40)
# example 6 reading lines from file line by line
camelot_lines = []
with open('raw_files/camelot.txt', 'r') as song:
    for line in song:
        camelot_lines.append(line.strip())
print(camelot_lines)