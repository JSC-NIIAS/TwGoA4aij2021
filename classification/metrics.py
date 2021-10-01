import torch
from sklearn.metrics import precision_score, f1_score

class Metrics_factory(object):
    def __init__(self,hyp,classes,model_type,loss_categories_mt,calc):
        self.classes_names=classes
        self.calc_type = hyp['validation']['calculation_type']
        self.sigmoid_threshold = hyp['validation']['sigmoid_treshold']
        self.model_type = model_type
        self.cats=loss_categories_mt
        self.calc=calc
    def compute_metrics(self,pred,targets):
        if self.calc_type == 'by_class':
            if self.model_type=='Single':
                f1_one_batch=[]
                for i in range(len(self.classes_names)):
                    f1 = f1_score((torch.sigmoid(pred[:,i]).data>self.sigmoid_threshold).to(torch.float32).cpu().detach().numpy(), targets[:,i].cpu().detach().numpy(), average='micro', zero_division=0)
                    f1_one_batch.append(f1)
            if self.model_type=='Multitask':
                indexes = []
                for categories in self.cats:
                    indexes_cat = []
                    for cat in categories:
                        indexes_cat.append(list(self.classes_names).index(cat))
                    indexes.append(indexes_cat)
                f1_one_batch=list(range(len(self.classes_names)))
                for i in range(len(self.calc)):
                    if self.calc[i]=='ML':
                        for index in indexes[i]:
                            f1=f1_score((torch.sigmoid(pred[i][:,indexes[i].index(index)]).data>self.sigmoid_threshold).to(torch.float32).cpu().detach().numpy(), targets[:,index].cpu().detach().numpy(), average='micro', zero_division=0)
                            f1_one_batch[index]=f1
                    if self.calc[i]=='MC':
                        indices = torch.tensor(indexes[i])
                        f1=f1_score(torch.argmax(torch.softmax(pred[i],dim=0),dim=1).to(torch.float32).cpu().detach().numpy(),torch.max(torch.index_select(targets.cpu(), 1, indices),dim=1)[1].detach().numpy(), average='micro', zero_division=0)
                        for index in indexes[i]:
                            f1_one_batch[index] = f1
        return f1_one_batch
