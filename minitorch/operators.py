"""
Mathematical operators for MiniTorch.

These form the foundation of all neural network operations.
You'll implement each function to understand how deep learning
frameworks handle basic mathematics.
"""

import math
from typing import Callable, Iterable


# TODO: Implement these functions in Task 0.1
def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def id(x: float) -> float:
    """Identity function."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x+y


def neg(x: float) -> float:
    """Negate a number."""
    return -x


def lt(x: float, y: float) -> float:
    """Less than comparison."""
    return 1.0 if x-y<0 else 0.0


def eq(x: float, y: float) -> float:
    """Equality comparison."""
    return 1.0 if x-y==0 else 0.0


def max(x: float, y: float) -> float:
    """Maximum of two numbers."""
    return x if x>y else y


def is_close(x: float, y: float) -> float:
    """Check if numbers are close."""
    return 1.0 if abs(x-y)<=1e-2 else 0.0

def sigmoid(x: float) -> float:
    """Sigmoid activation function."""
    if x>=0:
        result = 1.0/(1.0+(math.exp(-x)))
    else:
        expo=math.exp(-x)
        result = 1.0/(1.0+expo)
    if result==1:
        result=1-(1e-12)
    elif result==0:
        result=1e-12 
    return result


def relu(x: float) -> float:
    """ReLU activation function."""
    return x if x>0.0 else 0.0


def log(x: float) -> float:
    """Natural logarithm."""
    return math.log(x)


def exp(x: float) -> float:
    """Exponential function."""
    return math.exp(x)


def inv(x: float) -> float:
    """Reciprocal function."""
    return 1.0/x


def log_back(x: float, grad: float) -> float:
    """Gradient of log."""
    return grad/x


def inv_back(x: float, grad: float) -> float:
    """Gradient of inv."""
    return -grad/x**(2)


def relu_back(x: float, grad: float) -> float:
    """Gradient of ReLU."""
    return 0.0 if x<=0 else grad


# TODO: Implement these in Task 0.3
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    def mapped(ls: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in ls]
    return mapped

def zipWith(fn: Callable[[float, float], float]) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipWith function."""
    def zipped(ls1: Iterable[float],ls2: Iterable[float]) -> Iterable[float]:
        return [fn(x,y) for x,y in zip(ls1,ls2)]
    return zipped


def reduce(fn: Callable[[float, float], float], init: float) -> Callable[[Iterable[float]], float]:
    """Higher-order reduce function."""
    def reduced(ls: Iterable[float]) -> float:
        result=init
        for x in ls:
            result=fn(x,result)
        return result
    return reduced


def sum(ls: Iterable[float]) -> float:
    """Sum using reduce."""
    return reduce(add,0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Product using reduce."""
    return reduce(mul,1.0)(ls)


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate list using map."""
    return map(neg)(ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add lists using zipWith."""
    return zipWith(add)(ls1,ls2)

def mean(ls:Iterable[float]) -> float:
    ls=list(ls)
    sum=reduce(add,0.0)(ls)
    len=reduce(lambda x,y: y+1.0,0.0)(ls)
    return sum/len

