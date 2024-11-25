import torch

outputs = torch.tensor([[0.1,0.2],
                        [0.3,0.4]])

print(outputs.argmax(dim=1))
# print(outputs.argmax(dim=0))
preds = outputs.argmax(dim=1)
targets = torch.tensor([0,1])
print((preds==targets).sum())











