import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import CosineSimilarity
import numpy as np

class Ranking():
    def __init__(self, cast_feats, candidate_feats):
        self.cast_feats = cast_feats
        self.candidate_feats = candidate_feats

    def getCosine(self):
        
        all_feat = torch.cat((self.cast_feats, self.candidate_feats))
        
        #print(all_feat.shape)
        
        similarity = []
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
        for i in range(all_feat.shape[0]):
            feat = all_feat[i].expand(all_feat.shape[0], all_feat[0].shape[0])
            s = cos(feat, all_feat)
            
            s = np.array(s.cpu())
            similarity.append(s)
            
        similarity = np.array(similarity)

        return similarity
    
    def getCosine_train(self):
        
        all_feat = torch.cat((self.cast_feats, self.candidate_feats))
        
        #print(all_feat.shape)
        
        similarity = []
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
        for i in range(all_feat.shape[0]):
            feat = all_feat[i].expand(all_feat.shape[0], all_feat[0].shape[0])
            s = cos(feat, all_feat)
            
            similarity.append(s)
            
        similarity = torch.stack(similarity)

        return similarity
    
    
    def getRank(self):
        
        
        self.cast_feats = F.normalize(self.cast_feats, p=2, dim=1)
        self.candidate_feats = F.normalize(self.candidate_feats, p=2, dim=1)
        
        n_cast = self.cast_feats.shape[0]
        n_candidate = self.candidate_feats.shape[0]
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
        similarity = []
        indices = []
        for i in range(n_cast):
            cast_feat = self.cast_feats[i].expand(n_candidate, self.cast_feats[i].shape[0])
            s = cos(cast_feat, self.candidate_feats)
            
            s = s.cpu().numpy()
            
            idx = np.argsort(s)[::-1]
            similarity.append(s)
            indices.append(idx)
            
        return similarity, indices
    
    def getEuclideanRanking(self):
        
        self.cast_feats = F.normalize(self.cast_feats, p=2, dim=1)
        self.candidate_feats = F.normalize(self.candidate_feats, p=2, dim=1)
        
        m = self.cast_feats.shape[0]
        n = self.candidate_feats.shape[0]
        
        distmat = torch.pow(self.cast_feats, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(self.candidate_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()

        
        distmat.addmm_(1, -2, self.cast_feats, self.candidate_feats.t())
        distmat = distmat.cpu().numpy()
        
        dist = []
        indices = []
        for i in range(distmat.shape[0]):
            dist_i = distmat[i]
            idx = np.argsort(dist_i)
            indices.append(idx)   
         #   print(idx[0:10])
            
        return indices
    
    def getEuclideanRanking_train(self):
        
        self.cast_feats = F.normalize(self.cast_feats, p=2, dim=1)
        self.candidate_feats = F.normalize(self.candidate_feats, p=2, dim=1)
        
        m = self.cast_feats.shape[0]
        n = self.candidate_feats.shape[0]
        
        distmat = torch.pow(self.cast_feats, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(self.candidate_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()

        
        distmat.addmm_(1, -2, self.cast_feats, self.candidate_feats.t())
#         distmat = distmat.detach()
        
        return torch.argsort(distmat, dim=1)
#         dist = []
#         indices = []
#         for i in range(distmat.shape[0]):
#             dist_i = distmat[i]
#             idx = np.argsort(dist_i)
#             indices.append(idx)   
#          #   print(idx[0:10])
            
#         return indices