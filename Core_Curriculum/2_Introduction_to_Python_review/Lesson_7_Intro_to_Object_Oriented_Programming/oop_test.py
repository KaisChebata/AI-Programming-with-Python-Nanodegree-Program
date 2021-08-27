class A:
    def __init__(self, value):
        print('constructing A')
        self.x = value

class B(A):
    def __init__(self, value):
        print('constructing B')
        super().__init__(value)

    @property
    def x(self):
        print('getting x')
        return self._internalX

    @x.setter
    def x(self, new_x):
        print('setting x')
        self._internalX = new_x

class SomeClass:
    def __init__(self, value) -> None:
        print('constructing object ...')
        self.x = value
    
    @property
    def x(self):
        return self._x
    @x.setter
    def x(self, new_value):
        self._x = new_value

class P:
    def __init__(self, x) -> None:
        self.x = x
    
    @property
    def x(self):
        return self._x
    @x.setter
    def x(self, x):
        if x < 0:
            self._x = 0
        elif x > 1000:
            self._x = 1000
        else:
            self._x = x
    
    

# b = B('X')
# print('b.x = "{}"'.format(b.x))
# print('b._internalX = "{}"'.format(b._internalX))

# some = SomeClass('data')
# print(f'some = {some.x}')
# print(dir(SomeClass))
# some.x = 'd'
# print(some._x)

y = 15
x = 1 if y % 2 != 0 else 0
print(x)
