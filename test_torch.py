import torch

# Create two random matrices
a = torch.rand((3, 3))
b = torch.rand((3, 3))

print("Matrix A:\n", a)
print("Matrix B:\n", b)

# Matrix multiply
c = torch.matmul(a, b)
print("Result A x B:\n", c)

# Simple tensor operation
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
print("Dot product:", torch.dot(x, y))
