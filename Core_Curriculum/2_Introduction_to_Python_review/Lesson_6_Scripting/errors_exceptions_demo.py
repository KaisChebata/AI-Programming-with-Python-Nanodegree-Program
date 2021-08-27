# example1
# while True:
#     try:
#         x = int(input('Enter number: '))
#         break
#     except ValueError:
#         print('that is not valid number!')
#     except KeyboardInterrupt:
#         print('\nNo input taken!')
#         break
#     except:
#         print('run when any exceptions other than thos specified \
#             on the above except bloxk occures')
#     finally:
#         print('\nAttemped Input!\n')

# handling errors practice
def party_planner(cookies, people):
    left_overs = None
    num_each = None

    try:
        num_each = cookies // people
        left_overs = cookies % people
    except ZeroDivisionError as e:
        print('warning! enter a different number of people other than Zero.')
        print('ZeroDivisionError occured: {}'.format(e))
    
    return (num_each, left_overs)

# main code block
lets_party = 'y'
while lets_party == 'y':
    cookies = int(input('How many cookies are you baking? '))
    people = int(input("How many people are attending? "))

    cookies_each, leftovers = party_planner(cookies, people)

    if cookies_each:
        message = "\nLet's party! We'll have {} people attending, they'll each get to eat {} cookies, and we'll have {} left over."
        print(message.format(people, cookies_each, leftovers))
    
    lets_party = input('\nWould you like to party more? (y or n)').lower()
