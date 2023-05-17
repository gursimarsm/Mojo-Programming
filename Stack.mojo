from Pointer import Pointer
from IO import print_no_newline

struct Stack[T : AnyType]:
     var data: Pointer[T]
     var top : Int
     var size : Int
    
    fn __init__(inout self):
        self.top = -1
        self.size = 16
        self.data = Pointer[T].alloc(self.size)
        
    fn __init__(inout self,stackSize: Int):
        self.top = -1
        self.size = stackSize
        self.data = Pointer[T].alloc(self.size)
        
    fn isFull(inout self) -> Bool:
        return self.top == self.size
    
    fn push(inout self,value : T):
        if self.isFull():
            print('Stack is full')
            return
        self.top = self.top + 1
        self.data.store(self.top, value)
               
    fn pop(inout self) -> T :
        # if self.top == -1:
        #     print('Stack is empty')
        #     return None
        let popval = self.data[self.top]
        self.top = self.top - 1
        return popval
    
    fn __del__(owned self):
        self.data.free()
     
     
     
var s = Stack[Int]()
s.push(99)
s.push(98)
s.push(97)
print(s.pop())
print(s.pop())
print(s.pop())
        
    
