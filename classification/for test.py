import torch

pred=torch.rand([4,2]).float()
print(pred)
print(torch.argmax(torch.softmax(pred,dim=0),dim=1).cpu().detach().numpy())