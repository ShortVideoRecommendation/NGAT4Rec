import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import dgl
import pickle
import torch
import networkx as nx

from utility.Sampler import BiSampler, BiBlockSampler, HomoSampler, ComplexSampler     

class Data(object):
    def __init__(self, path, batch_size, multi_loss=0, use_attribute=0, sample_num_list=[100]):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        self.data_name = path.split('/')[-1]

        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}
        self.multi_loss = True if multi_loss == 1 else False
        self.use_attribute = True if use_attribute == 1 else False
        self.sample_num_list = sample_num_list

        self.exist_users = []

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1

        self.print_statistics()

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R_Item_Interacts = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)

        # self.g = dgl.DGLGraph()
        # self.g.add_nodes(self.n_users + self.n_items)

        self.train_items, self.test_set = {}, {}
        self.items_users = {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for idx, i in enumerate(train_items):
                        self.R[uid, i] = 1.
                        if i not in self.items_users.keys():
                            self.items_users[i] = [uid]
                        else:
                            self.items_users[i].append(uid)
                        # self.g.add_edge(uid, i + self.n_users)
                        # self.g.add_edge(i + self.n_users, uid)

                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

        self.R = self.R.tocsr()
        self.g_main = dgl.heterograph({
            ('user', 'u2i', 'item'): self.R,
            ('item', 'i2u', 'user'): self.R.transpose(),
        })

        self.u2i_graph = dgl.transform.metapath_reachable_graph(self.g_main, [('user', 'u2i', 'item')])
        self.i2u_graph = dgl.transform.metapath_reachable_graph(self.g_main, [('item', 'i2u', 'user')])

        if multi_loss:
            self.multi_loss_init()

        self.sampler = BiSampler(self.g_main, self.sample_num_list)
        self.block_sampler = BiBlockSampler(self.g_main, self.sample_num_list)

    def get_i2i_adj(self):
        R_i2i_user_side = self.R.transpose() * self.R
        R_i2i_item_side = self.R_item * self.R_item.transpose()

        self.i2i_user_side_lists = self.construct_adj(R_i2i_user_side, self.n_items)
        self.i2i_item_side_lists = self.construct_adj(R_i2i_item_side, self.n_items)
        self.i2i_all_lists = self.merge_constructed_adj(self.i2i_user_side_lists, self.i2i_item_side_lists)
        self.i2i_neg_lists = self.construct_neg_pool(self.i2i_all_lists, self.n_items)
        self.i2i_item_side_neg_lists = self.construct_neg_pool(self.i2i_item_side_lists, self.n_items)

    def construct_adj(self, csr_mat, n_element):
        adj_list = []
        for i in range(n_element):
            next_arr = np.asarray(
                csr_mat.indices[csr_mat.indptr[i]:csr_mat.indptr[i+1]])
            adj_list.append(next_arr)
        adj_list = np.asarray(adj_list)
        return adj_list

    def construct_neg_pool(self, adj_list, n_element):
        all_entities = list(range(n_element))
        neg_adj_list = []
        length = 0.
        for i in range(n_element):
            neg_list = np.setdiff1d(all_entities, adj_list[i])
            length += len(neg_list)
            neg_adj_list.append(neg_list)
        print('Negtive pool avg. length: {}'.format(length / n_element))
        return neg_adj_list

    def merge_constructed_adj(self, adj_list1, adj_list2):
        adj_list = []
        for list1, list2 in zip(adj_list1, adj_list2):
            adj_list.append(np.union1d(list1, list2))
        return adj_list
    
    def load_pkl(self, name):
        with open(self.path + name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def get_dgl_graph(self):
        if self.use_attribute:
            return self.g_main, self.g_info, self.attribute_cnt
        else:
            return self.g_main

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def get_D_inv(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            return d_mat_inv

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    def batch_sample(self):
        users, pos_items, neg_items = self.sample()
        pos_items_unique, pos_items_reverse = np.unique(pos_items, return_index=False, return_inverse=True)
        neg_items_unique, neg_items_reverse = np.unique(neg_items, return_index=False, return_inverse=True)
        return users, pos_items_unique, pos_items_reverse, neg_items_unique, neg_items_reverse

    def sample_inference_graph(self):
        users = torch.LongTensor(list(range(self.n_users)))
        items = torch.LongTensor(list(range(self.n_items)))
        graphs_list = self.block_sampler.sample(
            {'user': users, 'item': items})

        return graphs_list

    # for experiment on KS
    def sample_check_graph(self):
        users = torch.LongTensor([223, 265, 313, 320, 327, 355])
        items = torch.LongTensor([10793,10794,10795,10796])

        graphs_list = self.block_sampler.sample(
            {'user': users, 'item': items})
        return graphs_list

    # debugging
    def sample_inference_blocks(self):
        batch_num = 1000
        test_users = list(range(self.n_users))
        test_items = list(range(self.n_items))
        u_batch_size = self.n_users // batch_num
        i_batch_size = self.n_items // batch_num
        graphs_list = []
        for batch_id in range(batch_num):
            u_start = batch_id * u_batch_size
            u_end = (batch_id + 1) * u_batch_size

            i_start = batch_id * i_batch_size
            i_end = (batch_id + 1) * i_batch_size

            users = torch.LongTensor(test_users[u_start:u_end])
            items = torch.LongTensor(test_items[i_start:i_end])
            graphs_list.append(self.block_sampler.sample({'user': users, 'item': items}))
        return graphs_list

    def sample_block(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]
        # users = self.exist_users[:]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items


        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
            # neg_items += sample_neg_items_for_u(u, 3)
        # pos_items_unique = np.unique(pos_items, return_index=False, return_inverse=False)
        # neg_items_unique = np.unique(neg_items, return_index=False, return_inverse=False)
        items, items_reverse = np.unique(pos_items + neg_items, return_index=False, return_inverse=True)

        # items = np.concatenate([pos_items_unique, neg_items_unique])

        sample_seeds = {
            'user': torch.LongTensor(users),
            'item': torch.LongTensor(items),
        }
        sampled_graphs = self.block_sampler.sample(sample_seeds)

        # pos_item_graphs = self.sampler('item', pos_items_unique)
        # neg_item_graphs = self.sampler('item', neg_items_unique)

        sample_list = (users, pos_items, neg_items)
        reverse_list = (users, items_reverse[:self.batch_size], items_reverse[self.batch_size:])
        return sample_list, reverse_list, sampled_graphs

    def multi_loss_init(self):
        self.u2u_R = self.R * self.R.transpose()
        self.i2i_R = self.R.transpose() * self.R
        self.u2u_graph = dgl.graph(self.u2u_R)
        self.i2i_graph = dgl.graph(self.i2i_R)
        self.homo_sampler = HomoSampler(10)

    def sample_multi_loss(self):
        users = rd.sample(self.exist_users, self.batch_size)
        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
            # neg_items += sample_neg_items_for_u(u, 3)
        # pos_items_unique = np.unique(pos_items, return_index=False, return_inverse=False)
        # neg_items_unique = np.unique(neg_items, return_index=False, return_inverse=False)

        pos_items_unique = np.unique(pos_items)


        items = np.unique(pos_items + neg_items)



    def sample_graph(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]
        # users = self.exist_users[:]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items


        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
            # neg_items += sample_neg_items_for_u(u, 3)
        # pos_items_unique = np.unique(pos_items, return_index=False, return_inverse=False)
        # neg_items_unique = np.unique(neg_items, return_index=False, return_inverse=False)
        items = np.unique(pos_items + neg_items)

        # items = np.concatenate([pos_items_unique, neg_items_unique])

        sample_seeds = {
            'user': torch.LongTensor(users),
            'item': torch.LongTensor(items),
        }
        sampled_graphs = self.sampler.sample(sample_seeds)

        # pos_item_graphs = self.sampler('item', pos_items_unique)
        # neg_item_graphs = self.sampler('item', neg_items_unique)

        sample_list = (users, pos_items, neg_items)
        return sample_list, sampled_graphs

    def get_bipartite_graphs(self):
        return self.u2i_graph, self.i2u_graph
    
    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]
        # users = self.exist_users[:]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_pos_items_for_i(i, num):
            pos_items = self.i2i_item_side_lists[i]

            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch
        
        def sample_neg_items_for_i(i, num):
            neg_items = self.i2i_item_side_neg_lists[i]
            n_neg_items = len(neg_items)
            neg_batch = []
            while True:
                if len(neg_batch) == num: break
                neg_id = np.random.randint(low=0, high=n_neg_items, size=1)[0]
                neg_i_id = neg_items[neg_id]

                if neg_i_id not in neg_batch:
                    neg_batch.append(neg_i_id)
            return neg_batch

        def sample_neg_items_for_i_old(i, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.i2i_item_side_lists[i] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
            # neg_items += sample_neg_items_for_u(u, 3)

        if not self.multi_loss:
            return users, pos_items, neg_items
        else:
            pos_items_pos, pos_items_neg = [], []
            neg_items_pos, neg_items_neg = [], []
            for i in pos_items:
                pos_items_pos += sample_pos_items_for_i(i, 1)
                pos_items_neg += sample_neg_items_for_i(i, 1)
            for i in neg_items:
                neg_items_pos += sample_pos_items_for_i(i, 1)
                neg_items_neg += sample_neg_items_for_i(i, 1)
            return users, pos_items, neg_items, pos_items_pos, pos_items_neg, neg_items_pos, neg_items_neg
    
       
    def sample_batch_size(self, batch_size):
        if batch_size <= self.n_users:
            users = rd.sample(self.exist_users, batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(batch_size)]
        # users = self.exist_users[:]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_pos_items_for_i(i, num):
            pos_items = self.i2i_item_side_lists[i]

            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch
        
        def sample_neg_items_for_i(i, num):
            neg_items = self.i2i_item_side_neg_lists[i]
            n_neg_items = len(neg_items)
            neg_batch = []
            while True:
                if len(neg_batch) == num: break
                neg_id = np.random.randint(low=0, high=n_neg_items, size=1)[0]
                neg_i_id = neg_items[neg_id]

                if neg_i_id not in neg_batch:
                    neg_batch.append(neg_i_id)
            return neg_batch

        def sample_neg_items_for_i_old(i, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.i2i_item_side_lists[i] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
            # neg_items += sample_neg_items_for_u(u, 3)

        if not self.multi_loss:
            return users, pos_items, neg_items
        else:
            pos_items_pos, pos_items_neg = [], []
            neg_items_pos, neg_items_neg = [], []
            for i in pos_items:
                pos_items_pos += sample_pos_items_for_i(i, 1)
                pos_items_neg += sample_neg_items_for_i(i, 1)
            for i in neg_items:
                neg_items_pos += sample_pos_items_for_i(i, 1)
                neg_items_neg += sample_neg_items_for_i(i, 1)
            return users, pos_items, neg_items, pos_items_pos, pos_items_neg, neg_items_pos, neg_items_neg

    def sample_sym(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]
        # users = self.exist_users[:]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_users_for_i(i, num):
            neg_users = []
            while True:
                if len(neg_users) == num: break
                neg_id = np.random.randint(low=0, high=self.n_users, size=1)[0]
                if neg_id not in self.items_users[i] and neg_id not in neg_users:
                    neg_users.append(neg_id)
            return neg_users

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
        
        neg_users = []
        for i in pos_items:
            neg_users += sample_neg_users_for_i(i, 1)
            # neg_items += sample_neg_items_for_u(u, 3)

        return users, pos_items, neg_users, neg_items

    def sample_all_users_pos_items(self):
        self.all_train_users = []

        self.all_train_pos_items = []
        for u in self.exist_users:
            self.all_train_users += [u] * len(self.train_items[u])
            self.all_train_pos_items += self.train_items[u]

    def epoch_sample(self):
        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        neg_items = []
        for u in self.all_train_users:
            neg_items += sample_neg_items_for_u(u,1)

        perm = np.random.permutation(len(self.all_train_users))
        users = np.array(self.all_train_users)[perm]
        pos_items = np.array(self.all_train_pos_items)[perm]
        neg_items = np.array(neg_items)[perm]
        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))


    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state



    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)



        return split_uids, split_state

    def dataset_static(self):
        item_degrees = self.g_main.in_degrees(etype='u2i').detach().cpu().numpy()
        user_degrees = self.g_main.in_degrees(etype='i2u').detach().cpu().numpy()

        item_max = max(item_degrees)
        user_max = max(user_degrees)

        item_st = [0 for i in range(item_max // 50 + 1)]
        user_st = [0 for i in range(user_max // 50 + 1)]

        for i in item_degrees:
            item_st[i // 50] += 1
        for u in user_degrees:
            user_st[u // 50] += 1

        print('Item Max Degree:{}'.format(item_max))
        for i, item_s in enumerate(item_st):
            print('{}~{}:{}'.format(i*50, (i+1)*50, item_s))
        print('User Max Degree:{}'.format(user_max))
        for i, user_s in enumerate(user_st):
            print('{}~{}:{}'.format(i*50, (i+1)*50, user_s))

    def get_degree(self):
        # self.check_user = 69
        check_item1 = [5057,1708,473,5058,5059,621,3107,552]
        # self.check_user2 = 72
        check_item2 = [4091,2525,5128,2337,5129,5130,1463,5131,5132,5133,5134,692,3198]
        item_degrees = self.g_main.in_degrees(etype='u2i')
        print(item_degrees[check_item1])
        print(item_degrees[check_item2])



    
if __name__ == "__main__":
    data_generator = Data(path="Data/gowalla", batch_size=16)
    # g = data_generator.get_dgl_graph()
    # sub_g = g.edge_subgraph({('user', 'u2i', 'item'): [1, 2]})
    # print(sub_g.nodes['user'].data[dgl.NID])
    # print(sub_g.nodes['item'].data[dgl.NID])
    graph = data_generator.sample_triplet_graph()
    # a = torch.Tensor([1.,])
    # from ..Models import AsymModel
    # model = AsymModel(graph, data_generator.n_users, data_generator.n_items,
    #             2, a.device, u2i, i2u, 4, 1)
    # ua_embeddings, ia_embeddings = self.model()

