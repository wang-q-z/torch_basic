import  torch

device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32,device=device,requires_grad=True)
#how to initialize tensor
#print(my_tensor)
x = torch.empty(size= (3,3))
#print(x)
x = torch.zeros((3,3))
x = torch.rand((3,3))
x = torch.eye(5,5)
x =torch.arange(start=0, end=5,step=1)
x = torch.linspace(0.1,1,10)
x = torch.empty(size=(1,5)).normal_(mean=0,std=1)
x = torch.empty(size=(1,5)).uniform_(0,1)

x = torch.diag(torch.ones(3))


#how to initialize and convert tensors to other types(int float double)
tensor = torch.arange(4)
#print(tensor, tensor.bool())
print(tensor.short())
print(tensor.long())
print(tensor.half())
print(tensor.float())
print(tensor.double() )


#Array to tensor conversion and vice-versa
import numpy as np
np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
np_array_bacck = tensor.numpy()


#tensor math and comparison operations
x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

# addition
z1 = torch.empty(3)
torch.add(x,y,out=z1)
print("z1",z1)
z2 = torch.add(x,y)
z = x + y


#substraction
z = x -y

#division
z4 = torch.true_divide(x,y)
print(z4)

#inplace operations
t = torch.zeros(3)
t.add_(x)
t = t + x
print(t)

#Exponentiation

z = x.pow(2)
z = x**2


#simple comparison
z = x > 0

#matrix multiplication
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1,x2)
print("x3",x3)
x3 = x1.mm(x2)

#matrix exponentiation
matrix_exp = torch.rand(5,5)
matrix_exp.matrix_power(3)

#element wise mult
z = x  * y


#dot product
z = torch.dot(x,y)

#Batch Matrix mult
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch,n,m))
tensor2 = torch.rand((batch,m,p))
out_bmm = torch.bmm(tensor1,tensor2) #(batch n p)

#Example of Broadcasting
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))
z = x1 - x2
z = x1**x2
print(x1,x2,z)


#other useful tensor operations
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x,dim=0)
values, indices = torch.min(x,dim=0)
abs_x = torch.abs(x)
z = torch.argmax(x, dim=0)
mean_x = torch.mean(x.float(),dim=0)
z = torch.eq(x,y)
sorted_y,indices = torch.sort(y,dim=0,descending=False)

z = torch.clamp(x, min=0, max=10)#小于0的设为0

x = torch.tensor([1,0,1,1,1],dtype=torch.bool)
z = torch.any(x)
z = torch.all(x)

# tensor indexing

batch_size = 10
features = 25
x = torch.rand((batch_size,features))

print(x[0].shape) #x[0,:]
print(x[:,0].shape)

print(x[2,0:10])

#Fancy indexing
x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices])

x = torch.rand((3,5))
rows = torch.tensor([1,0])
cols = torch.tensor([4,0])
print(x[row,cols])

#more advanced indexing
x = torch.arange(10)


#tensor reshape
x = torch.arange(9)
x_3x3 = x.view(3,3)
x_3x3 = x.reshape(3,3)







