#from pytorch_metric_learning import losses
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable
from sklearn import metrics

"""
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feature_vectors, labels):
        # Normalize feature vectors

        feature_vectors_normalized = F.normalize(feature_vectors, p=2, dim=1)
        # Compute logits
        logits = torch.div(
            torch.matmul(
                feature_vectors_normalized, torch.transpose(feature_vectors_normalized, 0, 1)
            ),
            self.temperature,
        )
        #return losses.NTXentLoss(temperature=0.07)(logits, torch.squeeze(labels))
"""


def global_loss(cnn_code, rnn_code, eps=1e-8, temp3=10.0):

    batch_size = cnn_code.shape[0]
    print(f"batch_size: {batch_size}")
    labels = Variable(torch.LongTensor(range(batch_size))).to(cnn_code.device)
    print("labels")
    print(labels)

    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)

    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * temp3


    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    print(scores0)
    scores1 = scores0.transpose(0, 1)
    #scores1 = scores0
    loss0 = nn.CrossEntropyLoss()(scores0, labels)
    loss1 = nn.CrossEntropyLoss()(scores1, labels)
    return loss0, loss1


def get_global_similarities(img_emb_g, text_emb_g):
    img_emb_g = img_emb_g.detach().cpu().numpy()
    text_emb_g = text_emb_g.detach().cpu().numpy()
    global_similarities = metrics.pairwise.cosine_similarity(img_emb_g, text_emb_g)
    global_similarities = torch.Tensor(global_similarities)
    return global_similarities


class ContrastiveLoss(nn.Module):
    """Compute contrastive loss"""

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def sim(self, im, s):
        """Cosine similarity between all the image and sentence pairs"""
        return im.mm(s.t())

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > 0.5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()