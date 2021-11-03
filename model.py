import datetime
import math
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from scipy.sparse import coo_matrix
import time
import random
from numba import jit

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class ItemConv(Module):
    def __init__(self, layers, emb_size=100):
        super(ItemConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.w_item = {}
        for i in range(self.layers):
            self.w_item['weight_item%d' % (i)] = nn.Linear(self.emb_size, self.emb_size, bias=False)

    def forward(self, adjacency, embedding):
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        item_embeddings = embedding
        item_embedding_layer0 = item_embeddings
        final = [item_embedding_layer0]
        for i in range(self.layers):
            item_embeddings = trans_to_cuda(self.w_item['weight_item%d' % (i)])(item_embeddings)
            item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), item_embeddings)
            final.append(F.normalize(item_embeddings, dim=-1, p=2))
        item_embeddings = np.sum(final, 0)/(self.layers+1)
        return item_embeddings


class SessConv(Module):
    def __init__(self, layers, batch_size, emb_size=100):
        super(SessConv, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.layers = layers
        self.w_sess = {}
        for i in range(self.layers):
            self.w_sess['weight_sess%d' % (i)] = nn.Linear(self.emb_size, self.emb_size, bias=False)

    def forward(self, item_embedding, D, A, session_item, session_len):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros([1,self.emb_size])
        item_embedding = torch.cat([zeros, item_embedding], 0)
        seq_h = []
        for i in torch.arange(len(session_item)):
            seq_h.append(torch.index_select(item_embedding, 0, session_item[i]))
        seq_h1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in seq_h]))
        session_emb = torch.div(torch.sum(seq_h1, 1), session_len)
        session = [session_emb]
        DA = torch.mm(D, A).float()
        for i in range(self.layers):
            session_emb = trans_to_cuda(self.w_sess['weight_sess%d' % (i)])(session_emb)
            session_emb = torch.mm(DA, session_emb)
            session.append(F.normalize(session_emb, p=2, dim=-1))
        sess = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in session]))
        session_emb = torch.sum(sess, 0)/(self.layers+1)
        return session_emb


class COTREC(Module):
    def __init__(self, adjacency, n_node, lr, layers, l2, beta,lam,eps, dataset, emb_size=100, batch_size=100):
        super(COTREC, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.dataset = dataset
        self.L2 = l2
        self.lr = lr
        self.layers = layers
        self.beta = beta
        self.lam = lam
        self.eps = eps
        self.K = 10
        self.w_k = 10
        self.num = 5000
        self.adjacency = adjacency
        self.embedding = nn.Embedding(self.n_node, self.emb_size)
        self.pos_len = 200
        if self.dataset == 'retailrocket':
            self.pos_len = 300
        self.pos_embedding = nn.Embedding(self.pos_len, self.emb_size)
        self.ItemGraph = ItemConv(self.layers)
        self.SessGraph = SessConv(self.layers, self.batch_size)
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.emb_size, self.emb_size))
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.w_i = nn.Linear(self.emb_size, self.emb_size)
        self.w_s = nn.Linear(self.emb_size, self.emb_size)
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)

        self.adv_item = torch.cuda.FloatTensor(self.n_node, self.emb_size).fill_(0).requires_grad_(True)
        self.adv_sess = torch.cuda.FloatTensor(self.n_node, self.emb_size).fill_(0).requires_grad_(True)
        # self.adv_item = torch.zeros(self.n_node, self.emb_size).requires_grad_(True)
        # self.adv_sess = torch.zeros(self.n_node, self.emb_size).requires_grad_(True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def generate_sess_emb(self, item_embedding, session_item, session_len, reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros(1, self.emb_size)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)
        # seq_h = torch.zeros(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        hs = torch.div(torch.sum(seq_h, 1), session_len)
        mask = mask.float().unsqueeze(-1)
        len = seq_h.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)

        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, seq_h], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * seq_h, 1)
        return select

    def generate_sess_emb_npos(self, item_embedding, session_item, session_len, reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros(1, self.emb_size)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)
        # seq_h = torch.zeros(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        hs = torch.div(torch.sum(seq_h, 1), session_len)
        mask = mask.float().unsqueeze(-1)
        len = seq_h.shape[1]

        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.sigmoid(self.glu1(seq_h) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * seq_h, 1)
        return select

    def example_predicting(self, item_emb, sess_emb):
        x_u = torch.matmul(item_emb, sess_emb)
        pos = torch.softmax(x_u, 0)
        return pos

    def adversarial_item(self, item_emb, tar,sess_emb):
        adv_item_emb = item_emb + self.adv_item
        score = torch.mm(sess_emb, torch.transpose(adv_item_emb, 1, 0))
        loss = self.loss_function(score, tar)
        grad = torch.autograd.grad(loss, self.adv_item,retain_graph=True)[0]
        adv = grad.detach()
        self.adv_item = (F.normalize(adv, p=2,dim=1) * self.eps).requires_grad_(True)

    def adversarial_sess(self, item_emb, tar,sess_emb):
        adv_item_emb = item_emb + self.adv_sess
        score = torch.mm(sess_emb, torch.transpose(adv_item_emb, 1, 0))
        loss = self.loss_function(score, tar)
        grad = torch.autograd.grad(loss, self.adv_sess,retain_graph=True)[0]
        adv = grad.detach()
        self.adv_sess = (F.normalize(adv, p=2,dim=1) * self.eps).requires_grad_(True)

    def diff(self, score_item, score_sess, score_adv2, score_adv1, diff_mask):
        # compute KL(score_item, score_adv2), KL(score_sess, score_adv1)
        score_item = F.softmax(score_item, dim=1)
        score_sess = F.softmax(score_sess, dim=1)
        score_adv2 = F.softmax(score_adv2, dim=1)
        score_adv1 = F.softmax(score_adv1, dim=1)
        score_item = torch.mul(score_item, diff_mask)
        score_sess = torch.mul(score_sess, diff_mask)
        score_adv1 = torch.mul(score_adv1, diff_mask)
        score_adv2 = torch.mul(score_adv2, diff_mask)

        h1 = torch.sum(torch.mul(score_item, torch.log(1e-8 + ((score_item + 1e-8)/(score_adv2 + 1e-8)))))
        h2 = torch.sum(torch.mul(score_sess, torch.log(1e-8 + ((score_sess + 1e-8)/(score_adv1 + 1e-8)))))

        return h1+h2

    def SSL_topk(self, anchor, sess_emb, pos, neg):
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 2)

        anchor = F.normalize(anchor + sess_emb, p=2, dim=-1)
        pos = torch.reshape(pos, (self.batch_size, self.K, self.emb_size)) + sess_emb.unsqueeze(1).repeat(1, self.K, 1)
        neg = torch.reshape(neg, (self.batch_size, self.K, self.emb_size)) + sess_emb.unsqueeze(1).repeat(1, self.K, 1)
        pos_score = score(anchor.unsqueeze(1).repeat(1, self.K, 1), F.normalize(pos, p=2, dim=-1))
        neg_score = score(anchor.unsqueeze(1).repeat(1, self.K, 1), F.normalize(neg, p=2, dim=-1))
        pos_score = torch.sum(torch.exp(pos_score / 0.2), 1)
        neg_score = torch.sum(torch.exp(neg_score / 0.2), 1)
        con_loss = -torch.sum(torch.log(pos_score / (pos_score + neg_score)))
        return con_loss

    def topk_func_random(self, score1,score2, item_emb_I, item_emb_S):
        values, pos_ind_I = score1.topk(self.num, dim=0, largest=True, sorted=True)
        values, pos_ind_S = score2.topk(self.num, dim=0, largest=True, sorted=True)
        pos_emb_I = torch.cuda.FloatTensor(self.K, self.batch_size, self.emb_size).fill_(0)
        pos_emb_S = torch.cuda.FloatTensor(self.K, self.batch_size, self.emb_size).fill_(0)
        neg_emb_I = torch.cuda.FloatTensor(self.K, self.batch_size, self.emb_size).fill_(0)
        neg_emb_S = torch.cuda.FloatTensor(self.K, self.batch_size, self.emb_size).fill_(0)
        for i in torch.arange(self.K):
            pos_emb_S[i] = item_emb_S[pos_ind_I[i]]
            pos_emb_I[i] = item_emb_I[pos_ind_S[i]]
        random_slices = torch.randint(self.K, self.num, (self.K,))  # choose negative items
        for i in torch.arange(self.K):
            neg_emb_S[i] = item_emb_S[pos_ind_I[random_slices[i]]]
            neg_emb_I[i] = item_emb_I[pos_ind_S[random_slices[i]]]
        return pos_emb_I, neg_emb_I, pos_emb_S, neg_emb_S

    def forward(self, session_item, session_len, D, A, reversed_sess_item, mask, epoch, tar, train, diff_mask):
        if train:
            item_embeddings_i = self.ItemGraph(self.adjacency, self.embedding.weight)
            if self.dataset == 'Tmall':
                # for Tmall dataset, we do not use position embedding to learn temporal order
                sess_emb_i = self.generate_sess_emb_npos(item_embeddings_i, session_item, session_len,reversed_sess_item, mask)
            else:
                sess_emb_i = self.generate_sess_emb(item_embeddings_i, session_item, session_len, reversed_sess_item, mask)
            sess_emb_i = self.w_k * F.normalize(sess_emb_i, dim=-1, p=2)
            item_embeddings_i = F.normalize(item_embeddings_i, dim=-1, p=2)
            scores_item = torch.mm(sess_emb_i, torch.transpose(item_embeddings_i, 1, 0))
            loss_item = self.loss_function(scores_item, tar)

            sess_emb_s = self.SessGraph(self.embedding.weight, D, A, session_item, session_len)
            scores_sess = torch.mm(sess_emb_s, torch.transpose(item_embeddings_i, 1, 0))
            # compute probability of items to be positive examples
            pos_prob_I = self.example_predicting(item_embeddings_i, sess_emb_i)
            pos_prob_S = self.example_predicting(self.embedding.weight, sess_emb_s)

            # choose top-10 items as positive samples and randomly choose 10 items as negative and get their embedding
            pos_emb_I, neg_emb_I, pos_emb_S, neg_emb_S = self.topk_func_random(pos_prob_I,pos_prob_S, item_embeddings_i, self.embedding.weight)

            last_item = torch.squeeze(reversed_sess_item[:, 0])
            last_item = last_item - 1
            last = item_embeddings_i.index_select(0, last_item)
            con_loss = self.SSL_topk(last, sess_emb_i, pos_emb_I, neg_emb_I)
            last = self.embedding(last_item)
            con_loss += self.SSL_topk(last, sess_emb_s, pos_emb_S, neg_emb_S)

            # compute and update adversarial examples
            self.adversarial_item(item_embeddings_i, tar, sess_emb_i)
            self.adversarial_sess(item_embeddings_i, tar, sess_emb_s)

            adv_emb_item = item_embeddings_i + self.adv_item
            adv_emb_sess = item_embeddings_i + self.adv_sess

            score_adv1 = torch.mm(sess_emb_s, torch.transpose(adv_emb_item, 1, 0))
            score_adv2 = torch.mm(sess_emb_i, torch.transpose(adv_emb_sess, 1, 0))
            # add difference constraint
            loss_diff = self.diff(scores_item, scores_sess, score_adv2, score_adv1, diff_mask)
        else:
            item_embeddings_i = self.ItemGraph(self.adjacency, self.embedding.weight)
            if self.dataset == 'Tmall':
                sess_emb_i = self.generate_sess_emb_npos(item_embeddings_i, session_item, session_len, reversed_sess_item, mask)
            else:
                sess_emb_i = self.generate_sess_emb(item_embeddings_i, session_item, session_len, reversed_sess_item, mask)
            sess_emb_i = self.w_k * F.normalize(sess_emb_i, dim=-1, p=2)
            item_embeddings_i = F.normalize(item_embeddings_i, dim=-1, p=2)
            scores_item = torch.mm(sess_emb_i, torch.transpose(item_embeddings_i, 1, 0))
            loss_item = self.loss_function(scores_item, tar)
            loss_diff = 0
            con_loss = 0
        return self.beta * con_loss, loss_item, scores_item, loss_diff*self.lam


def forward(model, i, data, epoch, train):
    tar, session_len, session_item, reversed_sess_item, mask, diff_mask = data.get_slice(i)
    diff_mask = trans_to_cuda(torch.Tensor(diff_mask).long())
    A_hat, D_hat = data.get_overlap(session_item)
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    A_hat = trans_to_cuda(torch.Tensor(A_hat))
    D_hat = trans_to_cuda(torch.Tensor(D_hat))
    tar = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    con_loss, loss_item, scores_item, loss_diff = model(session_item, session_len, D_hat, A_hat, reversed_sess_item, mask, epoch,tar, train, diff_mask)
    return tar, scores_item, con_loss, loss_item, loss_diff

@jit(nopython=True)
def find_k_largest(K, candidates):
    n_candidates = []
    for iid,score in enumerate(candidates[:K]):
        n_candidates.append((iid, score))
    n_candidates.sort(key=lambda d: d[1], reverse=True)
    k_largest_scores = [item[1] for item in n_candidates]
    ids = [item[0] for item in n_candidates]
    # find the N biggest scores
    for iid,score in enumerate(candidates):
        ind = K
        l = 0
        r = K - 1
        if k_largest_scores[r] < score:
            while r >= l:
                mid = int((r - l) / 2) + l
                if k_largest_scores[mid] >= score:
                    l = mid + 1
                elif k_largest_scores[mid] < score:
                    r = mid - 1
                if r < l:
                    ind = r
                    break
        # move the items backwards
        if ind < K - 2:
            k_largest_scores[ind + 2:] = k_largest_scores[ind + 1:-1]
            ids[ind + 2:] = ids[ind + 1:-1]
        if ind < K - 1:
            k_largest_scores[ind + 1] = score
            ids[ind + 1] = iid
    return ids#,k_largest_scores


def train_test(model, train_data, test_data, epoch):
    print('start training: ', datetime.datetime.now())
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i in slices:
        model.zero_grad()
        tar, scores_item, con_loss, loss_item, loss_diff = forward(model, i, train_data, epoch, train=True)
        loss = loss_item + con_loss + loss_diff
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()
    print('\tLoss:\t%.3f' % total_loss)
    top_K = [5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
    print('start predicting: ', datetime.datetime.now())

    model.eval()
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        tar,scores_item, con_loss, loss_item, loss_diff = forward(model, i, test_data, epoch, train=False)
        scores = trans_to_cpu(scores_item).detach().numpy()
        index = []
        for idd in range(model.batch_size):
            index.append(find_k_largest(20, scores[idd]))
        index = np.array(index)
        tar = trans_to_cpu(tar).detach().numpy()
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                metrics['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
    return metrics, total_loss


