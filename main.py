import time
from util import Data
from model import *
import os
import argparse
import pickle
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Tmall', help='dataset name: retailrocket/diginetica/Nowplaying/sample')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--embSize', type=int, default=100, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=float, default=2, help='the number of layer used')
parser.add_argument('--beta', type=float, default=0.005, help='ssl task maginitude')
parser.add_argument('--lam', type=float, default=0.005, help='diff task maginitude')
parser.add_argument('--eps', type=float, default=0.2, help='eps')

opt = parser.parse_args()
print(opt)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# torch.cuda.set_device(1)

def main():
    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
    all_train = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
    if opt.dataset == 'diginetica':
        n_node = 43097
    elif opt.dataset == 'Tmall':
        n_node = 40727
    elif opt.dataset == 'retailrocket':
        n_node = 36968
    else:
        n_node = 309
    train_data = Data(train_data,all_train, shuffle=True, n_node=n_node)
    test_data = Data(test_data,all_train, shuffle=True, n_node=n_node)
    model = trans_to_cuda(COTREC(adjacency=train_data.adjacency,n_node=n_node,lr=opt.lr, l2=opt.l2, beta=opt.beta,lam= opt.lam,eps=opt.eps,layers=opt.layer,emb_size=opt.embSize, batch_size=opt.batchSize,dataset=opt.dataset))
    top_K = [5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        metrics, total_loss = train_test(model, train_data, test_data, epoch)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
        print(metrics)
        for K in top_K:
            print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tEpoch: %d,  %d' %
                  (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))


if __name__ == '__main__':
    main()
