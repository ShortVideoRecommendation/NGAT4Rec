import numpy as np
import random as rd
import networkx as nx
import scipy.sparse as sp
from time import time

class SparseDatasetGenerator(object):
    def __init__(self, path='kuaishou'):
        self.path = path
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0

        self.train_items, self.test_set = {}, {}
        self.u2i_dataset = {}
        self.i2u_dataset = {}
        # with open(train_file) as f_train:
        #     with open(test_file) as f_test:
        #         for l in f_train.readlines():
        #             if len(l) == 0: break
        #             l = l.strip('\n')
        #             items = [int(i) for i in l.split(' ')]
        #             uid, train_items = items[0], items[1:]

        #             for i in train_items:
        #                 self.R[uid, i] = 1.
                        
        #             self.train_items[uid] = train_items
        self.user_deg_list = []
        self.item_deg_list = []

        with open(train_file) as f_train:
            for l in f_train.readlines():
                if len(l) == 0: break
                l = l.strip('\n')
                items = [int(i) for i in l.split(' ')]
                uid, train_items = items[0], items[1:]
                self.n_items = max(self.n_items, max(items))
                self.n_users = max(self.n_users, uid)
                self.u2i_dataset[uid] = train_items
                for train_item in train_items:
                    if train_item not in self.i2u_dataset.keys():
                        self.i2u_dataset[train_item] = [uid]
                    else:
                        self.i2u_dataset[train_item].append(uid)
                self.n_train += len(train_items)
        
        with open(test_file) as f_test:
            for l in f_test.readlines():
                if len(l) == 0: break
                l = l.strip()
                items = [int(i) for i in l.split(' ')]
                # try:
                #     items = [int(i) for i in l.split(' ')]
                # except:
                #     print(l)
                uid, test_items = items[0], items[1:]
                self.u2i_dataset[uid].extend(test_items)
                for test_item in test_items:
                    self.i2u_dataset[test_item].append(uid)
                self.n_test += len(test_items)


        for user_id in range(self.n_users):
            self.user_deg_list.append(len(self.u2i_dataset[user_id]))
        for item_id in range(self.n_items):
            self.item_deg_list.append(len(self.i2u_dataset[item_id]))

        self.n_items += 1
        self.n_users += 1
        self.n_edges = self.n_train + self.n_test
        self.users_avg_degree = self.n_edges / self.n_users
        self.items_avg_degree = self.n_edges / self.n_items

        self.old_density = self.n_edges / (self.n_users * self.n_items)
        self.users_mid_degree = self.user_deg_list[self.n_users // 2]
        self.items_mid_degree = self.item_deg_list[self.n_items // 2]
        # self.sparse_dataset = {}

        print('users avg degree: {:5f}, items avg degree: {:.5f}'.format(self.users_avg_degree, self.items_avg_degree))

        self.graph = nx.Graph()
        for user_id in range(self.n_users):
            items = self.u2i_dataset[user_id]
            if len(items) < self.users_avg_degree:
            # if len(items) < self.users_mid_degree:
                for item_id in items:
                    if len(self.i2u_dataset[item_id]) < self.items_avg_degree:
                        self.graph.add_edge('u' + str(user_id), 'i' + str(item_id))
        
        self.graph_5_core = nx.k_core(self.graph, k=5)
        self.new_dataset = nx.to_dict_of_lists(self.graph_5_core)

        train_dataset = dict()
        test_dataset = dict()
        train_dataset_reid = dict()
        test_dataset_reid = dict()
        users_dict = dict()
        items_dict = dict()
        users_cnt = 0
        items_cnt = 0
        index = 0
        n_train_sparse = 0
        n_test_sparse = 0

        for k, v in self.new_dataset.items():
            if k[0] == 'u':
                train_num = (len(v) * 4) // 5
                train_dataset[k] = v[:train_num]
                test_dataset[k] = v[train_num:]

                n_train_sparse += len(train_dataset[k])
                n_test_sparse += len(test_dataset[k])

                if k not in users_dict.keys():
                    users_dict[k] = users_cnt
                    users_cnt += 1
                k_reid = users_dict[k]
                v_reid = []

                for d in v:
                    if d not in items_dict.keys():
                        items_dict[d] = items_cnt
                        items_cnt += 1
                    
                    v_reid.append(str(items_dict[d]))

                train_dataset_reid[k_reid] = v_reid[:train_num]
                test_dataset_reid[k_reid] = v_reid[train_num:]

                if index % 1000 == 0:
                    print('Current:', index)
                index += 1

        self.new_density = (n_train_sparse + n_test_sparse) / (users_cnt * items_cnt)
        print('old density: {:.4f}, new density: {:.4f}'.format(self.old_density, self.new_density))
        if self.old_density > self.new_density:
            with open("kuaishou_sparse/train.txt", "w") as train_file:
                for k, v in train_dataset_reid.items():
                    new_str = str(k) + ' ' + ' '.join(v) + '\n'
                    train_file.write(new_str)
            with open("kuaishou_sparse/test.txt", "w") as test_file:
                for k, v in test_dataset_reid.items():
                    new_str = str(k) + ' ' + ' '.join(v) + '\n'
                    test_file.write(new_str)
            with open("kuaishou_sparse/train_original.txt", "w") as train_file:
                for k, v in train_dataset.items():
                    new_str = str(k) + ' ' + ' '.join(v) + '\n'
                    train_file.write(new_str)
            with open("kuaishou_sparse/test_original.txt", "w") as test_file:
                for k, v in test_dataset.items():
                    new_str = str(k) + ' ' + ' '.join(v) + '\n'
                    test_file.write(new_str)
            with open("kuaishou_sparse/users_list.txt", "w") as ul_file:
                for k, v in users_dict.items():
                    new_str = str(k) + ' ' + str(v) + '\n'
                    ul_file.write(new_str)
            with open("kuaishou_sparse/items_list.txt", "w") as il_file:
                for k, v in items_dict.items():
                    new_str = str(k) + ' ' + str(v) + '\n'
                    il_file.write(new_str)

        
if __name__ == '__main__':
    generator = SparseDatasetGenerator()

        
        

        


