class Shirt:
    def __init__(self, shirt_color, shirt_size, shirt_style, shirt_price):
        self.color = shirt_color
        self.size = shirt_size
        self.style = shirt_style
        self.price = shirt_price
    
    @property
    def price(self):
        return self._price
    
    @price.setter
    def price(self, new_price):
        self._price = new_price

    # def change_price(self, new_price):
    #     self.price = new_price
    
    def discount(self, discount):
        return self.price * (1 - discount)

new_shirt = Shirt('red', 'S', 'short sleeve', 15)
print(f'new_shirt attributes:')
print(f'color: {new_shirt.color}')
print(f'size: {new_shirt.size}')
print(f'style: {new_shirt.style}')
print(f'price: {new_shirt.price}')
new_shirt.price = 10
print('*' * 10)
print(new_shirt._price)
print(f'change new_shirt price to: ${new_shirt.price}')
print(f'price after discount of 20%: {new_shirt.discount(.2)}')
print('-' * 40)
tshirt_collection = []
shirt_one = Shirt('orange', 'M', 'short sleeve', 25)
shirt_two = Shirt('red', 'S', 'short sleeve', 15)
shirt_three = Shirt('purple', 'XL', 'short sleeve', 10)
tshirt_collection.extend([shirt_one, shirt_two, shirt_three])

for i in range(len(tshirt_collection)):
    print(tshirt_collection[i].color)