import os
import sys
import math
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn
import dgl.function as fn
from dgl.nn.pytorch.softmax import edge_softmax

from utility.helper import *
from utility.batch_test_hetero import *

#===============================Convolution Defination===============================
class GraphAttentionConv(nn.Module):
    def __init__(self, emb_dim, negative_slope=0.2, **kwargs):
        super(GraphAttentionConv, self).__init__()
        self.emb_dim = emb_dim
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def message_func(self, edges):
        return {
            'src_emb': edges.data['src_emb'],
            'e': edges.data['e'],
            # 'er': edges.data['er'],
        }

    def reduce_func(self, nodes):
        embs = nodes.mailbox['src_emb'] # [A, #n_neighbor, dim]
        e = nodes.mailbox['e'] # [A, #n_neighbor, 1]
        # print(embs.shape)
        a = torch.softmax(self.leaky_relu(e), dim=1) # [A, #n_neighbor, 1]
        # print(a.shape)
        out_embs = torch.sum(a * embs, dim=1)
        # print(out_embs.shape)

        return {'emb': out_embs}

    def forward(self, G, feat_dict):
        user_src_embs = feat_dict['user']
        item_src_embs = feat_dict['item']

        # user_src_norm_embs = F.normalize(user_src_embs, p=2, dim=1)
        # item_src_norm_embs = F.normalize(item_src_embs, p=2, dim=1)

        number_of_dst_user = G.number_of_dst_nodes('user')
        number_of_dst_item = G.number_of_dst_nodes('item')
        
        user_dst_embs = feat_dict['user'][:number_of_dst_user]
        item_dst_embs = feat_dict['item'][:number_of_dst_item]

        G.srcnodes['user'].data.update({
            'src_emb': user_src_embs,
        })
        G.dstnodes['user'].data.update({
            'dst_emb': user_dst_embs,
        })

        G.srcnodes['item'].data.update({
            'src_emb': item_src_embs,
        })
        G.dstnodes['item'].data.update({
            'dst_emb': item_dst_embs,
        })

        msg_embs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # G.apply_edges(fn.copy_u('src_emb', 'src_emb'), etype=etype)
            # G.apply_edges(fn.u_add_v('el', 'er', 'e'), etype=etype)
            
            # G.apply_edges(fn.copy_e('e', 'e'), etype=etype)
            # G.apply_edges(fn.copy_v('er', 'er'), etype=etype)

            # G.update_all(message_func=self.message_func, reduce_func=self.reduce_func, etype=etype)
            # msg_embs[dsttype] = G.dstnodes[dsttype].data['emb']
            G.apply_edges(fn.u_dot_v('src_emb', 'dst_emb', 'e'), etype=etype)
            G.edges[etype].data['e'] = G.edges[etype].data.pop('e').view(-1, 1)
            G.apply_edges(fn.copy_u('src_emb', 'src_emb'), etype=etype)
            G.apply_edges(fn.copy_e('e', 'e'), etype=etype)

            G.update_all(message_func=self.message_func, reduce_func=self.reduce_func, etype=etype)
            msg_embs[dsttype] = G.dstnodes[dsttype].data['emb']

        return msg_embs

#===============================Model Defination===============================
class LightGATIP(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, 
                 n_layers, batch_size, decay, *args, **kwargs):
        super(LightGATIP, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.decay = decay

        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)

        self._init_weight_()

        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(GraphAttentionConv(self.emb_dim))

    def _init_weight_(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def inference(self, blocks_list):
        # print(blocks_list)
        user_ids = blocks_list[0].srcnodes['user'].data[dgl.NID]
        item_ids = blocks_list[0].srcnodes['item'].data[dgl.NID]

        feat_dict = {
            'user': self.user_embedding.weight[user_ids],
            'item': self.item_embedding.weight[item_ids],
        }

        number_of_centric_user = blocks_list[-1].number_of_dst_nodes('user')
        number_of_centric_item = blocks_list[-1].number_of_dst_nodes('item')

        user_embeddings = [feat_dict['user'][:number_of_centric_user]]
        item_embeddings = [feat_dict['item'][:number_of_centric_item]]

        for block, layer in zip(blocks_list, self.layers):
            feat_dict = layer(block, feat_dict)
            user_embeddings.append(feat_dict['user'][:number_of_centric_user])
            item_embeddings.append(feat_dict['item'][:number_of_centric_item])
        
        user_embeddings = torch.stack(user_embeddings, dim=1)
        user_embeddings = torch.mean(user_embeddings, dim=1)

        item_embeddings = torch.stack(item_embeddings, dim=1)
        item_embeddings = torch.mean(item_embeddings, dim=1)

        return user_embeddings, item_embeddings

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1./2*(users.norm(2).pow(2) + pos_items.norm(2).pow(2) + neg_items.norm(2).pow(2))
        regularizer = regularizer / self.batch_size

        mf_loss = torch.mean(F.softplus(neg_scores - pos_scores))

        emb_loss = self.decay * regularizer
        reg_loss = 0
        return mf_loss, emb_loss, reg_loss

    def forward(self, graph_list, pos_items_ur, neg_items_ur):
        user_embeddings, item_embeddings = self.inference(graph_list)
        batch_users = user_embeddings
        batch_pos_items = item_embeddings[pos_items_ur]
        batch_neg_items = item_embeddings[neg_items_ur]
        mf_loss, emb_loss, reg_loss = self.bpr_loss(
            batch_users, batch_pos_items, batch_neg_items)

        return mf_loss, emb_loss, reg_loss


#===============================Model Wrapper===============================
class Model_Wrapper(object):
    def __init__(self, data_config, pretrain_data):
        self.model_type = args.model_type
        self.adj_type = args.adj_type
        self.alg_type = 'LightGATIP'

        self.mess_dropout = eval(args.mess_dropout)

        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.record_alphas = False

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.neighbors_num = eval(args.neighbors_num)
        self.n_layers = len(self.neighbors_num)
        self.sample_test_flag=args.sample_test_flag

        # assert self.n_layers == len(self.neighbors_num), ""

        self.model_type += '_%s_l%d' % (self.alg_type, self.n_layers)

        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.verbose = args.verbose

        print('model_type is {}'.format(self.model_type))

        self.weights_save_path = '%sweights/%s/%s/d%s/l%s/r%s' % (args.weights_path, args.dataset, self.model_type, str(self.emb_dim),
                                                                 str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))
        self.pretrain_weights_save_path = '%sweights/%s/%s/d%s/l%s/r%s' % (args.weights_path, args.dataset, self.model_type, str(self.emb_dim),
                                                                 str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))

        self.g = data_generator.get_dgl_graph()
        self.u2i_graph, self.i2u_graph = data_generator.get_bipartite_graphs()

        """
        *********************************************************
        Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
        """

        print('----self.alg_type is {}----'.format(self.alg_type))
        self.model = LightGATIP(
            self.n_users, self.n_items,
            self.emb_dim, self.n_layers,
            self.batch_size, self.decay)
        self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.lr_scheduler = self.set_lr_scheduler()
        print(self.model)
        for name, param in self.model.named_parameters():
            print(name, ' ', param.size())

    def set_lr_scheduler(self):
        fac = lambda epoch: 0.96 ** (epoch / 50)
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        return scheduler

    def save_model(self):
        ensureDir(self.weights_save_path)
        torch.save(self.model.state_dict(), self.weights_save_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.pretrain_weights_save_path))

    def test(self, users_to_test, drop_flag=False, batch_test_flag=False):
        self.model.eval()
        with torch.no_grad():
            # if self.sample_test_flag == 'full':
            #     test_graph_dict_list = [{'u2i':self.g, 'i2u':self.g} for _ in range(self.n_layers)]
            # else:
            #     test_graph_dict_list = data_generator.sample_inference_graph()
            # if self.sample_test_flag == 'full':
            #     test_graph_list = [self.g for _ in range(self.n_layers)]
            # else:
            test_graph_list = data_generator.sample_inference_graph()
            # print(test_graph_list)
            ua_embeddings, ia_embeddings = self.model.inference(test_graph_list)
            # print(ua_embeddings.shape)
            # print(ia_embeddings.shape)
            ua_embeddings = ua_embeddings.detach()
            ia_embeddings = ia_embeddings.detach()
        # result = test_torch(ua_embeddings, ia_embeddings, users_to_test)
        result = test_torch(ua_embeddings, ia_embeddings, users_to_test)
        return result

    def check_train(self):
        self.model.eval()
        with torch.no_grad():
            check_graph_list = data_generator.sample_check_graph()
            Eu, Ei = self.model.inference(check_graph_list)
            norm_Eu = torch.norm(Eu, dim=1, p=2).cpu().numpy()
            norm_Ei = torch.norm(Ei, dim=1, p=2).cpu().numpy()
            print('l2norm of Eu:', norm_Eu)
            print('Eu:', Eu.cpu().numpy())
            print('l2norm of Ei', norm_Ei)
            print('Ei:', Ei.cpu().numpy())

    def train(self):
        training_time_list = []
        loss_loger, pre_loger, rec_loger, ndcg_loger = [], [], [], []
        stopping_step = 0
        should_stop = False
        cur_best_pre_0 = 0.

        n_batch = data_generator.n_train // args.batch_size + 1
        batch_split = [args.batch_size for i in range(3)]

        for epoch in range(args.epoch):
            t1 = time()
            loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
            f_time, b_time, loss_time, opt_time, clip_time, emb_time = 0., 0., 0., 0., 0., 0.
            sample_time = 0.
            cuda_time = 0.

            for idx in range(n_batch):
                self.model.train()
                self.optimizer.zero_grad()
                sample_t1 = time()
                
                sample_list, reverse_list, graph_list = data_generator.sample_block()
                users, pos_items, neg_items = sample_list
                _, pos_items_reverse, neg_items_reverse = reverse_list
                sample_time += time() - sample_t1

                batch_mf_loss, batch_emb_loss, batch_reg_loss = self.model(
                    graph_list, pos_items_reverse, neg_items_reverse)
                batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss

                batch_loss.backward()
                self.optimizer.step()

                loss += float(batch_loss)
                mf_loss += float(batch_mf_loss)
                emb_loss += float(batch_emb_loss)
                reg_loss += float(batch_reg_loss)


            self.lr_scheduler.step()

            if math.isnan(loss) == True:
                print('ERROR: loss is nan.')
                sys.exit()
            # ================================Start================================
            # self.check_train()
            # =================================End=================================
            # print the test evaluation metrics each 10 epochs
            if (epoch + 1) % 1 != 0 :
                if args.verbose > 0 and epoch % args.verbose == 0:
                    perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                        epoch + 1, time() - t1, loss, mf_loss, emb_loss)
                    training_time_list.append(time() - t1)
                    print(perf_str)
                continue

            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            ret = self.test(users_to_test, drop_flag=True)
            training_time_list.append(t2 - t1)

            t3 = time()

            loss_loger.append(loss)
            rec_loger.append(ret['recall'])
            pre_loger.append(ret['precision'])
            ndcg_loger.append(ret['ndcg'])
            if args.verbose > 0:
                perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                           'precision=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                           (epoch + 1, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, ret['recall'][0],
                            ret['recall'][-1],
                            ret['precision'][0], ret['precision'][-1],
                            ret['ndcg'][0], ret['ndcg'][-1])
                print(perf_str)
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=50)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            if should_stop:
                break

            # *********************************************************
            # save the user & item embeddings for pretraining.
            if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
                # save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
                self.save_model()
                if self.record_alphas:
                    self.best_alphas = [i for i in self.model.get_alphas()]
                print('save the weights in path: ', self.weights_save_path)

        if rec_loger != []:
            self.print_final_results(rec_loger, pre_loger, ndcg_loger, training_time_list)

    def print_final_results(self, rec_loger, pre_loger, ndcg_loger, training_time_list):
        recs = np.array(rec_loger)
        pres = np.array(pre_loger)
        ndcgs = np.array(ndcg_loger)

        best_rec_0 = max(recs[:, 0])
        idx = list(recs[:, 0]).index(best_rec_0)

        final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], ndcg=[%s]" % \
                     (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                      '\t'.join(['%.5f' % r for r in pres[idx]]),
                      '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
        print(final_perf)

        # Benchmarking: time consuming
        avg_time = sum(training_time_list) / len(training_time_list)
        time_consume = "Benchmarking time consuming: average {}s per epoch".format(avg_time)
        print(time_consume)

        results_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, self.model_type)

        ensureDir(results_path)
        f = open(results_path, 'a')

        f.write(
            'datetime: %s\nembed_size=%d, lr=%.5f, mess_dropout=%s, regs=%s, adj_type=%s\n\t%s\n\t%s\n\n'
            % (datetime.datetime.now(), args.embed_size, args.lr, args.mess_dropout, args.regs,
               args.adj_type, final_perf, time_consume))
        f.close()

def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data


#===============================Main===============================
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """

    t0 = time()

    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None

    Engine = Model_Wrapper(data_config=config, pretrain_data=pretrain_data)
    if args.pretrain:
        print('pretrain path: ', Engine.pretrain_weights_save_path)
        if os.path.exists(Engine.pretrain_weights_save_path):
            Engine.load_model()
            users_to_test = list(data_generator.test_set.keys())
            ret = Engine.test(users_to_test, drop_flag=True)
            cur_best_pre_0 = ret['recall'][0]

            pretrain_ret = 'pretrained model recall=[%.5f, %.5f], precision=[%.5f, %.5f],' \
                           'ndcg=[%.5f, %.5f]' % \
                           (ret['recall'][0], ret['recall'][-1],
                            ret['precision'][0], ret['precision'][-1],
                            ret['ndcg'][0], ret['ndcg'][-1])
            print(pretrain_ret)
        else:
            print('Cannot load pretrained model. Start training from stratch')
    else:
        print('without pretraining')
    Engine.train()


