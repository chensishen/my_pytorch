import torch
import torch.nn

inputs = torch.tensor([1,2,3],dtype=torch.float)
targets = torch.tensor([1,2,5],dtype=torch.float)

inputs = torch.reshape(inputs,(1,1,1,3))
targets = torch.reshape(targets,(1,1,1,3))

loss1 = torch.nn.L1Loss(reduction='sum')
result = loss1(inputs,targets)

print(result)

loss2 = torch.nn.MSELoss()
result = loss2(inputs,targets)

print(result)


loss2 = torch.nn.MSELoss()
result = loss2(inputs,targets)

print(result)

x = torch.tensor([0.1,0.2,0.3],dtype=torch.float)
y = torch.tensor([1])
x = torch.reshape(x,(1,3))
loss3 = torch.nn.CrossEntropyLoss()
result = loss3(x,y)

print(result)



