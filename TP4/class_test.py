class pepe:
    def __init__(self,v0):
        self.v0 = v0

    def dx(self,t,x):
        return self.v0 + t + x


def fun(f):
    print(f(1,2))

pepe(0)

fun(pepe.dx)
