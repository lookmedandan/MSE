import torch
import torch.nn as nn
class CrossEntropyBoundSmoothLoss(nn.Module):
    def __init__(self, e, d, bound_ids, batch_size, seq_len, label_num, device='cuda'):
        super().__init__()
        self.e = e
        self.d = d
        self.bound_ids = bound_ids
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.label_num = label_num
        self.device = device
    def forward(self, logits, label_ids):
        log_probs = self.logsoftmax(logits)  #[batch, seq_len, num_label]
        targets = torch.zeros(log_probs.size()).to(self.device).scatter_(1, label_ids.unsqueeze(1), 1) #[batch*seqlen, num_label]

        split_size = []
        for i in range(self.batch_size):
            split_size.append(self.seq_len)
        split_size = tuple(split_size)
        splits = torch.split(targets, split_size, dim=0)

        index = torch.LongTensor(self.bound_ids).to(self.device)
        bound_mask = torch.zeros(self.label_num, self.seq_len).to(self.device)
        bound_mask = bound_mask.index_fill(0, index, 1)
        smoothed_splits = []
        for i in range(len(splits)):
            label_tensor = splits[i].clone().t()
            masked_tensor = label_tensor*bound_mask
            indexs = torch.nonzero(masked_tensor).to('cpu').numpy()

            ones_tensor = torch.ones(masked_tensor.size()).to(self.device)
            zeros_tensor = torch.zeros(masked_tensor.size()).to(self.device)
            reversed_mask = torch.where(bound_mask==1, zeros_tensor, ones_tensor)
            reversed_mask = label_tensor*reversed_mask
            for ind in indexs:
                low = max(0, ind[1]-self.d)
                high = min(masked_tensor.size(1)-1, ind[1]+self.d)
                for j in range(low, high+1):
                    if j!=ind[1]:
                        masked_tensor[ind[0], j] = self.e/(high - low)
                masked_tensor[ind[0],ind[1]] = 1-self.e
            smoothed_splits.append((masked_tensor + reversed_mask).t())
        smoothed_splits = torch.stack(tuple(smoothed_splits), dim=0)
        smoothed_splits = smoothed_splits.view(-1,self.label_num) 
        loss=(-smoothed_splits*log_probs).mean(0).sum()
        return loss

class CrossEntropyBoundSmoothLoss_ScaleAverage(nn.Module):
    def __init__(self, e, d, bound_ids, batch_size, seq_len, label_num, device='cuda'):
        super().__init__()
        self.e = e
        self.d = d
        self.bound_ids = bound_ids
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.label_num = label_num
        self.device = device
    def forward(self, logits, label_ids):
        log_probs = self.logsoftmax(logits)
        targets = torch.zeros(log_probs.size()).to(self.device).scatter_(1, label_ids.unsqueeze(1), 1) #[batch*seqlen, num_label]

        split_size = []
        for i in range(self.batch_size):
            split_size.append(self.seq_len)
        split_size = tuple(split_size)
        splits = torch.split(targets, split_size, dim=0)

        index = torch.LongTensor(self.bound_ids).to(self.device)
        bound_mask = torch.zeros(self.label_num, self.seq_len).to(self.device)
        bound_mask = bound_mask.index_fill(0, index, 1)
        smoothed_splits = []
        for i in range(len(splits)):
            label_tensor = splits[i].clone().t()
            masked_tensor = label_tensor*bound_mask
            indexs = torch.nonzero(masked_tensor).to('cpu').numpy()

            ones_tensor = torch.ones(masked_tensor.size()).to(self.device)
            zeros_tensor = torch.zeros(masked_tensor.size()).to(self.device)
            reversed_mask = torch.where(bound_mask==1, zeros_tensor, ones_tensor)
            reversed_mask = label_tensor*reversed_mask
            for ind in indexs:
                for d in range(self.d, 0, -1):
                    count = 0
                    low = ind[1]-d
                    high = ind[1]+d
                    if low>=0:
                        count += 1
                    if high<=masked_tensor.size(1)-1:
                        count += 1
                    if low>=0:
                        masked_tensor[ind[0], low] = self.e/(self.d*count)
                    if high<=masked_tensor.size(1)-1:
                        masked_tensor[ind[0], high] = self.e/(self.d*count)
                masked_tensor[ind[0],ind[1]] = 1-self.e
            smoothed_splits.append((masked_tensor + reversed_mask).t()) #[seq_len, num_label]
        smoothed_splits = torch.stack(tuple(smoothed_splits), dim=0)   #[batch, seq_len, num_label]
        smoothed_splits = smoothed_splits.view(-1,self.label_num) 
        loss=(-smoothed_splits*log_probs).mean(0).sum()
        return loss

