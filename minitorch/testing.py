"""Testing utilities for MiniTorch."""

def assert_close(a: float, b: float, eps: float = 1e-2):
    """Assert two floats are close within tolerance."""
    assert abs(a - b) < eps, f"Values not close: {a} vs {b}"
from minitorch.operators import sigmoid, relu

print(f"sigmoid(0) = {sigmoid(0)}")
print(f"sigmoid(100) = {sigmoid(100)}")
print(f"sigmoid(-100) = {sigmoid(-100)}")
print(f"relu(-5) = {relu(-5)}")
print(f"relu(5) = {relu(5)}")
