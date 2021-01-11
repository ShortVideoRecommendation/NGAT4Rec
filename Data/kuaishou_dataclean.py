import sys
import time
from collections import defaultdict
import numpy as np
import networkx as nx
import pandas as pd

if __name__ == "__main__":
    # input_files = ["KSupa/1.csv", "KSupa/2.csv",
    #                "KSupa/3.csv", "KSupa/4.csv"]
    input_files = ["KSupa/1.csv"]

    graph = nx.Graph()

    dataset = dict()
    items_reid = dict()
    next_item_id = 0
    users_reid = dict()
    next_user_id = 0

    user_item_dict = {}
    # example:
    # user_item_dict = {
    #     "user1": {
    #         "item1": [is_click,is_like,is_comment,is_long_view]
    #     },
    # }
    #
    #

    for input_file in input_files:
        with open(input_file, "r") as f:
            next(f)
            for edge in f.readlines():
                if edge[-3] != "]":
                    # item_ids = item_ids[:-1]
                    print("end of file {}".format(input_file))
                    continue
                
                nodes = edge.rstrip("\n").split(",")
                user_id = 'u' + nodes[0].strip()
                # item_ids = edge[len(user_id)+3:-3].rstrip("\n").split(",")
                behaviors = edge[len(user_id)+2:].rstrip("\n").split("[")
                # for b in behaviors:
                #     print(b)
                # print(user_id)
                # print(item_ids)
                item_ids = behaviors[0][:-4].strip().split(',')
                # print(item_ids)
                # if len(item_ids) < 2:
                #     print(edge)
                #     continue
                is_click_list = behaviors[1][:-4].strip().split(',')
                is_like_list = behaviors[2][:-4].strip().split(',')
                is_comment_list = behaviors[3][:-4].strip().split(',')
                is_long_view_list = behaviors[4][:-2].strip().split(',')
                # print(item_ids)
                # print(is_click_list)
                # print(is_like_list)
                # print(is_comment_list)
                # print(is_long_view_list)

                assert len(item_ids)==len(is_click_list), "len(item_ids)!=len(is_click_list)"
                assert len(item_ids)==len(is_like_list), "len(item_ids)!=len(is_like_list)"
                assert len(item_ids)==len(is_comment_list), "len(item_ids)!=len(is_comment_list)"
                assert len(item_ids)==len(is_long_view_list), "len(item_ids)!=len(is_long_view_list)"

                user_item_dict.update({user_id:dict()})
                for i in range(len(item_ids)):
                    item_id = 'i' + item_ids[i]
                    user_item_dict[user_id].update({
                        item_id: [
                            is_click_list[i], is_like_list[i],
                            is_comment_list[i], is_long_view_list[i],
                        ]
                    })

                    graph.add_edge(user_id, item_id)
    
    graph_5_core = nx.k_core(graph, k=1)
    # all_nodes = graph_10_core.nodes()
    new_dataset = nx.to_dict_of_lists(graph_5_core)

    train_dataset = dict()
    test_dataset = dict()
    train_dataset_reid = dict()
    test_dataset_reid = dict()
    users_dict = dict()
    items_dict = dict()
    users_cnt = 0
    items_cnt = 0
    index = 0
    # print(user_item_dict['u2939904']['i34924618583'][1])
    for k, v in new_dataset.items():
        if k[0] == 'u':
            if len(v) < 2:
                print(k, v)
                continue
            train_num = (len(v) * 4) // 5
            train_dataset[k] = v[:train_num]
            test_dataset[k] = v[train_num:]

            if k not in users_dict.keys():
                users_dict[k] = users_cnt
                users_cnt += 1
            k_reid = users_dict[k]
            v_reid = []
            v_like_list = []
            v_comment_list = []
            v_long_view_list = []
            for d in v:
                if d not in items_dict.keys():
                    items_dict[d] = items_cnt
                    items_cnt += 1
                
                v_reid.append(str(items_dict[d]))
                v_like_list.append(str(user_item_dict[k][d][1]))
                v_comment_list.append(str(user_item_dict[k][d][2]))
                v_long_view_list.append(str(user_item_dict[k][d][3]))
            
            train_dataset_reid[k_reid] = [
                v_reid[:train_num],
                # v_like_list[:train_num],
                # v_comment_list[:train_num], 
                # v_long_view_list[:train_num],
            ]
            test_dataset_reid[k_reid] = [
                v_reid[train_num:], 
                # v_like_list[train_num:],
                # v_comment_list[train_num:], 
                # v_long_view_list[train_num:],
            ]

            if index % 1000 == 0:
                print('Current:', index)
            index += 1

    with open("KSupa/train.txt", "w") as train_file:
        for k, v in train_dataset_reid.items():
            new_str_list = [str(k)]
            for behavior_list in v:
                new_str_list.append(' '.join(behavior_list))
            # new_str = str(k) + ' ' + ' '.join(v) + '\n'
            new_str = ' '.join(new_str_list) + '\n'
            # print(new_str)
            # print('====')
            train_file.write(new_str)
    with open("KSupa/test.txt", "w") as test_file:
        for k, v in test_dataset_reid.items():
            new_str_list = [str(k)]
            for behavior_list in v:
                new_str_list.append(' '.join(behavior_list))
            # new_str = str(k) + ' ' + ' '.join(v) + '\n'
            new_str = ' '.join(new_str_list) + '\n'
            test_file.write(new_str)
    # with open("KSupa/train_original.txt", "w") as train_file:
    #     for k, v in train_dataset.items():
    #         new_str = str(k) + ' ' + ' '.join(v) + '\n'
    #         train_file.write(new_str)
    # with open("KSupa/test_original.txt", "w") as test_file:
    #     for k, v in test_dataset.items():
    #         new_str = str(k) + ' ' + ' '.join(v) + '\n'
    #         test_file.write(new_str)
    with open("KSupa/users_list.txt", "w") as ul_file:
        for k, v in users_dict.items():
            new_str = str(k) + ' ' + str(v) + '\n'
            ul_file.write(new_str)
    with open("KSupa/items_list.txt", "w") as il_file:
        for k, v in items_dict.items():
            new_str = str(k) + ' ' + str(v) + '\n'
            il_file.write(new_str)


                # if user_id in users_reid.keys():
                #     new_user_id = users_reid[user_id]
                # else:
                #     new_user_id = next_user_id
                #     next_user_id += 1
                #     users_reid[user_id] = new_user_id

                # for item_id in item_ids:
                #     if item_id in items_reid.keys():
                #         new_item_id = items_reid[item_id]
                #     else:
                #         new_item_id = next_item_id
                #         next_item_id += 1
                #         items_reid[item_id] = new_item_id

                #     if new_user_id in dataset.keys():
                #         dataset[new_user_id].append(str(new_item_id))
                #     else:
                #         dataset[new_user_id] = [str(new_item_id)]
    
    # train_dataset = dict()
    # test_dataset = dict()
    # index = 0
    # for user, items in dataset.items():
    #     if (len(items) < 5):
    #         print("userid:{}".format(user))
    #     train_num = (len(items) * 4) // 5
    #     train_dataset[user] = items[:train_num]
    #     test_dataset[user] = items[train_num:]
    #     if index % 1000 == 0:
    #         print('Current:', index)
    #     index += 1

    
    # with open("KSupa/train.txt", "w") as train_file:
    #     for k, v in train_dataset.items():
    #         new_str = str(k) + ' ' + ' '.join(v) + '\n'
    #         train_file.write(new_str)
    # with open("KSupa/test.txt", "w") as test_file:
    #     for k, v in test_dataset.items():
    #         new_str = str(k) + ' ' + ' '.join(v) + '\n'
    #         test_file.write(new_str)
    # with open("KSupa/users_list.txt", "w") as ul_file:
    #     for k, v in users_reid.items():
    #         new_str = str(k) + ' ' + str(v) + '\n'
    #         ul_file.write(new_str)
    # with open("KSupa/items_list.txt", "w") as il_file:
    #     for k, v in items_reid.items():
    #         new_str = str(k) + ' ' + str(v) + '\n'
    #         il_file.write(new_str)
    


