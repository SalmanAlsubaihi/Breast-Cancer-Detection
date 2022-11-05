from torch import nn

class ViewLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cosine_similarity = nn.CosineEmbeddingLoss()
    def __call__(self, out, _):
        out_L_CC, out_R_CC, out_L_MLO, out_R_MLO = out
        out_L_CC_extended, out_R_CC_extended = torch.cat([out_L_CC, out_L_CC.roll(1,0)]), torch.cat([out_R_CC, out_R_CC])
        out_L_MLO_extended, out_R_MLO_extended = torch.cat([out_L_MLO, out_L_MLO]), torch.cat([out_R_MLO, out_R_MLO.roll(1,0)])
        ones = torch.ones_like(out_L_CC[:,0])
        target = torch.cat([ones,-1*ones])
        
        loss1 = self.cosine_similarity(out_L_CC_extended, out_L_MLO_extended, target)
        loss2 = self.cosine_similarity(out_R_CC_extended, out_R_MLO_extended, target)
        loss3 = self.cosine_similarity(out_L_CC_extended, out_R_CC_extended, target)
        loss4 = self.cosine_similarity(out_L_MLO_extended, out_R_MLO_extended, target)
        ## view prediction loss
        loss = loss1 + loss2 + loss3 + loss4
        return loss
        
loss_functions = {
    'cross_entropy':nn.CrossEntropyLoss,
    'view_loss':ViewLoss
}