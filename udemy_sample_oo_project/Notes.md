creating class will create a new type i.e. type of Class, so book type, polynomial type
encapsulation : state (prop, variable) + behaviour (method)

inheritance : create new classes using super

abstraction : user of the class does not need to know how the class is implemented, same thing achieved by functions also

polymorphism : one thing many forms

class is also called as class object
instances created from class object is instance object

interface : all methods that client calling application need to worry about; contrast to 
private methods that are for internal working of Class

private variables (to be used only inside class) should start with _ e.g. self._x=x

-----------------------------
by getter and setter we are changing the property of class instances
calling functions should not access class variables on their own and need to go via getter and setter:

in the below code value is a instance variable

getter by using @property
    @property
    def value(self):
        return self._x


setter format:
    @value.setter
    def value(self,val):
        self._x=val

______________________________
if only getter provided without setter then it is a read only property
to delete the property use this decorator
@value.deleter
def value(self):
	print('value is deleted')
del p.value

