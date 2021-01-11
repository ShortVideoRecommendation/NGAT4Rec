import sys
import time
import numpy as np
import networkx as nx

if __name__ == "__main__":
    t = time.time()
    input_file = "Data/kuaishou.csv"

    graph = nx.Graph()

    with open(input_file, "r") as file:
        for edge in file:
            nodes = edge.rstrip("\n").split(',')
            user = 'u' + nodes[0].strip()
            items = nodes[1].strip().split(' ')

            for item in items:
                graph.add_edge(user, 'i' + item)
    
    graph_10_core = nx.k_core(graph, k=10)
    # all_nodes = graph_10_core.nodes()
    new_dataset = nx.to_dict_of_lists(graph_10_core)

    # index = 0
    # new_dataset_dict = dict()
    # for k, vs in dataset_dict.items():
    #     if k in all_nodes:
    #         new_items_list = []
    #         item_10_core_neighbors = nx.all_neighbors(graph_10_core, k)
    #         for v in vs:
    #             if v in item_10_core_neighbors:
    #                 new_items_list.append(v)
    #         new_dataset_dict[k] = new_items_list
    #         if index % 1000 == 0:
    #             print('current:', index)
    #         index += 1
        
    # with open("Data/dataset_10.txt", "w") as file:
    #     for u, ilist in new_dataset.items():
    #         if len(u) < 11:
    #             new_str = u + ' ' + ' '.join(ilist) + '\n'
    #             file.write(new_str)

    train_dataset = dict()
    test_dataset = dict()
    train_dataset_reid = dict()
    test_dataset_reid = dict()
    users_dict = dict()
    items_dict = dict()
    users_cnt = 0
    items_cnt = 0
    index = 0
    for k, v in new_dataset.items():
        if k[0] == 'u':
            train_num = (len(v) * 4) // 5
            train_dataset[k] = v[:train_num]
            test_dataset[k] = v[train_num:]

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

    with open("nebula/train.txt", "w") as train_file:
        for k, v in train_dataset_reid.items():
            new_str = str(k) + ' ' + ' '.join(v) + '\n'
            train_file.write(new_str)
    with open("nebula/test.txt", "w") as test_file:
        for k, v in test_dataset_reid.items():
            new_str = str(k) + ' ' + ' '.join(v) + '\n'
            test_file.write(new_str)
    with open("nebula/train_original.txt", "w") as train_file:
        for k, v in train_dataset.items():
            new_str = str(k) + ' ' + ' '.join(v) + '\n'
            train_file.write(new_str)
    with open("nebula/test_original.txt", "w") as test_file:
        for k, v in test_dataset.items():
            new_str = str(k) + ' ' + ' '.join(v) + '\n'
            test_file.write(new_str)
    with open("nebula/users_list.txt", "w") as ul_file:
        for k, v in users_dict.items():
            new_str = str(k) + ' ' + str(v) + '\n'
            ul_file.write(new_str)
    with open("nebula/items_list.txt", "w") as il_file:
        for k, v in items_dict.items():
            new_str = str(k) + ' ' + str(v) + '\n'
            il_file.write(new_str)

    print(time.time() - t, 's')

