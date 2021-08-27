'''
write a program tha ask the user to enter 10 two-digit numbers. 
It should then find and print the sum of all of the even numbers 
among those that were entered. 

Sample Output: This is what the output should look like.

>>> user_list: [23, 24, 25, 26, 27, 28, 29, 30, 31, 22]
>>> The sum of the even numbers in user_list is: 130.

'''

user_list = []
list_sum = 0

for i in range(10):
    user_input = int(input('Enter any 2-digit number: '))

    try:
        number = user_input
        user_list.append(number)
        if number % 2 == 0:
            list_sum += number
    except ValueError:
        print("Incorrect value. That's not an int!")

print(f'user_list: {user_list}')
print(f'The sum of the even numbers in user_list is: {list_sum}')