class Geeks:

    avar = 10
    
    @property
    def avar(cls):
        print("getter method called")
        return cls.avar
    
    @avar.setter
    def avar(cls, a):
        if(a < 18):
            raise ValueError("Sorry you age is below eligibility criteria")
        print("setter method called")
        cls.avar = a
    

    
    def __init__(self):
        self._age = 0
       
    # using property decorator
    # a getter function
    @property
    def age(self):
        print("getter method called")
        return self._age
       
    # a setter function
    @age.setter
    def age(self, a):
        if(a < 18):
            raise ValueError("Sorry you age is below eligibility criteria")
        print("setter method called")
        self._age = a
