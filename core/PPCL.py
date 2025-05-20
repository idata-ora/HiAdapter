import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from pdb import set_trace as st

class PPCL(torch.nn.Module):
    def __init__(self, num_classes=6, embedding=512, epoch=0, mrg=0, alpha=32, pos_loss_weight=0.5, neg_loss_weight=0.1, args=None):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.num_classes = num_classes
        self.embedding = embedding
        self.epoch = epoch
        self.mrg = mrg
        self.alpha = alpha
        self.pos_loss_weight = pos_loss_weight
        self.neg_loss_weight = neg_loss_weight
        self.iter = 0
        self.args = args
        self.save_path = os.path.join(self.args.save_proxies, 'proxies')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if self.epoch == 0:
            # self.proxies = torch.nn.Parameter(torch.ones(self.num_classes, self.embedding).cuda())
            self.proxies = torch.nn.Parameter(torch.randn(self.num_classes, self.embedding).cuda())
            nn.init.kaiming_normal_(self.proxies, mode='fan_out')
            print('=> init proxies')
        else:
            self.proxies = torch.load(os.path.join(self.save_path, str(self.epoch-1) + '.pt')).cuda()
            self.proxies = nn.Parameter(self.proxies)
            print(f'=> loaded proxies from {self.epoch-1}.pt')

    @torch.no_grad()
    def init_proxies(self,features,labels,cls_scores):
        assert features.shape[0] == labels.shape[0] == cls_scores.shape[0]
        cls_scores = F.softmax(cls_scores, dim=1) if cls_scores is not None else None
        cls_scores = cls_scores.cpu()
        labels = labels.cpu()
        features = features.cpu()
        self.class_label = torch.arange(0, self.num_classes)  # .cuda()
        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)
        label_mask = torch.eq(labels, self.class_label.T).float() # [batch_size, num_classes]
        score_mask = torch.mul(label_mask,cls_scores).T # [num_classes, batch_size]
        score_mask = torch.where(score_mask > 0.8, score_mask, torch.zeros_like(score_mask)) ##todo
        score_mask = score_mask.unsqueeze(dim=-1).repeat([1,1,self.embedding]) # [num_classes, batch_size, 768]
        features = features.unsqueeze(dim=0).repeat([self.num_classes, 1, 1])
        proxies = torch.mul(score_mask, features)
        proxies = F.normalize(proxies.sum(dim=1), dim=-1)
        proxies = torch.add(proxies, self.proxies.cpu())

        self.proxies = nn.Parameter(F.normalize(proxies, dim=-1).cuda())
        if self.iter == self.args.iternum:
            torch.save(self.proxies, os.path.join(self.save_path, str(self.epoch) + '.pt'))
            # print(f'saved proxies => {self.epoch}.pt')
            # print(self.proxies)
            self.iter = 0
        else:
            self.iter = self.iter + 1

    def forward(self, x, labels):
        # st()
        assert x.shape[0] == labels.shape[0]
        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)
        P = F.normalize(self.proxies, dim=-1)
        x = F.normalize(x, dim=-1)
        cos = F.linear(x, P)  # Calcluate cosine similarity
        class_label = torch.arange(0, self.num_classes).cuda()
        P_one_hot = torch.eq(labels, class_label.T).float().cuda() 
        N_one_hot = 1 - P_one_hot
        #exp
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)  # The set of positive proxies of data in the batch

        P_sim_max = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).max(dim=1).values
        N_sim_max = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).max(dim=1).values

        

        pos_term = torch.log(1 + P_sim_max).sum() / len(P_sim_max)  #  1+
        neg_term = torch.log(1 + N_sim_max).sum() / len(N_sim_max)
        # loss = pos_term + neg_term

        return self.pos_loss_weight*pos_term, self.neg_loss_weight*neg_term 