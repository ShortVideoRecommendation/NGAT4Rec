import dgl
import torch

class BiBlockSampler(object):
    def __init__(self, g, sample_num_list):
        self.g = g
        self.sample_num_list = sample_num_list

    def sample(self, dstnodes):
        """
        Input: dstnodes: {'user': torch.Tensor(), 'item': torch.Tensor()}
        """
        cur_dstnodes = dstnodes
        sampled_graphs = []

        for index, sample_num in enumerate(self.sample_num_list):
            sampled_graph = dgl.sampling.sample_neighbors(self.g, cur_dstnodes, sample_num)

            sampled_block = dgl.to_block(sampled_graph, cur_dstnodes, include_dst_in_src=True)
            sampled_graphs.insert(0, sampled_block)
            cur_dstnodes = {
                'user': sampled_block.srcnodes['user'].data[dgl.NID],
                'item': sampled_block.srcnodes['item'].data[dgl.NID],
            }

        return sampled_graphs

class ComplexSampler(object):
    def __init__(self, g_main, g_support, sample_num_list):
        self.g_main = g_main
        self.g_support = g_support
        self.sample_num_list = sample_num_list

    def sample(self, dstnodes):
        """
        Input: dstnodes: {'user': torch.Tensor(), 'item': torch.Tensor()}
        """
        cur_dstnodes = dstnodes
        sampled_graphs = []

        for sample_num in self.sample_num_list:
            sampled_graph_main = dgl.sampling.sample_neighbors(
                self.g_main, cur_dstnodes, sample_num)
            sampled_block_main = dgl.to_block(
                sampled_graph_main, cur_dstnodes, include_dst_in_src=True)

            cur_dstnodes = {
                'user': sampled_block.srcnodes['user'].data[dgl.NID],
                'item': sampled_block.srcnodes['item'].data[dgl.NID],
            }

            sampled_graph_support = dgl.sampling.sample_neighbors(
                self.g_support, cur_dstnodes, sample_num)
            sampled_block_support = dgl.to_block(
                sampled_graph_support, cur_dstnodes, include_dst_in_src=False)

            sampled_graphs.insert(0, (sampled_graph_main, sampled_graph_support))

        return sampled_graphs

class BiSampler(object):
    def __init__(self, g, sample_num_list):
        self.g = g
        self.sample_num_list = sample_num_list
    
    def get_block_srcnodes(self, g, dst_nodes):
        dst_nodes = {
            ntype: dgl.utils.toindex(nodes).tousertensor()
            for ntype, nodes in dst_nodes.items()}
        
        dst_nodes_nd = []
        for ntype in g.ntypes:
            nodes = dst_nodes.get(ntype, None)
            if nodes is not None:
                dst_nodes_nd.append(dgl.backend.zerocopy_to_dgl_ndarray(nodes))
            else:
                dst_nodes_nd.append(dgl.ndarray.NULL["int64"])
        print(dst_nodes_nd)
        _, src_nodes_nd, _ = dgl.transform._CAPI_DGLToBlock(g._graph, dst_nodes_nd, False)

        srcnodes = {}
        print(int(src_nodes_nd[0]))
        for i, ntype in enumerate(g.ntypes):
            srcnodes[ntype] = dgl.backend.zerocopy_from_dgl_ndarray(src_nodes_nd[i])

        return srcnodes

    def sample(self, dstnodes):
        """
        Input: dstnodes: {'user': torch.Tensor(), 'item': torch.Tensor()}
        """
        sampled_graphs = []
        cur_dstnodes = dstnodes
        # print(dstnodes)

        for index, sample_num in enumerate(self.sample_num_list):
            # 采样的中心节点是采样子图里的目标节点（有点绕）
            # sampled_i2u_graph = dgl.sampling.sample_neighbors(
            #     self.g['i2u'], {'user': cur_seed_dict['user']}, sample_num)
            # sampled_u2i_graph = dgl.sampling.sample_neighbors(
            #     self.g['u2i'], {'item': cur_seed_dict['item']}, sample_num)
            sampled_graph = dgl.sampling.sample_neighbors(self.g, cur_dstnodes, sample_num)
            sampled_graphs.insert(0, sampled_graph)

            if index == len(self.sample_num_list) - 1:
                break

            # srcnodes = self.get_block_srcnodes(sampled_graph, cur_dstnodes)
            sampled_block = dgl.to_block(sampled_graph, cur_dstnodes)

            # print(sampled_block.nodes['user'].data[dgl.NID])
            # print(sampled_block.nodes['item'].data[dgl.NID])

            cur_dstnodes = {
                'user': sampled_block.nodes['user'].data[dgl.NID],
                'item': sampled_block.nodes['item'].data[dgl.NID],
            }

        return sampled_graphs



class HomoSampler(object):
    def __init__(self, sample_num):
        self.sample_num = sample_num

    def sample(self, g, dstnodes):
        """
        Input: dstnodes: torch.Tensor()
        """
        sampled_graphs = []
        
        sampled_graph = dgl.sampling.sample_neighbors(g, dstnodes, self.sample_num)
        sampled_block = dgl.to_block(sampled_graph, dstnodes, include_dst_in_src=False)
        return sampled_block


class BipartiteSampler(object):
    def __init__(self, g, sample_num_list):
        self.g = g
        self.sample_num_list = sample_num_list
        self.next_dict = {
            g.ntypes[0]: g.ntypes[1],
            g.ntypes[1]: g.ntypes[0]}

    def sample(self, ntype, seeds):
        seeds = torch.LongTensor(seeds)
        sampled_graphs = []
        curr_ntype = ntype
        for index, sample_num in enumerate(self.sample_num_list):
            # print(curr_ntype)
            # print(seeds.shape)
            sampled_graphs.append(dgl.sampling.sample_neighbors(self.g, {curr_ntype: seeds}, sample_num))
            # print(self.g)
            # print(sampled_graphs[-1])
            if index == len(self.sample_num_list) - 1:
                break
            
            sampled_block = dgl.to_block(sampled_graphs[-1], {curr_ntype: seeds}, include_dst_in_src=False)
            curr_ntype = self.next_dict[curr_ntype]
                
            seeds = sampled_block.nodes[curr_ntype].data[dgl.NID]
        return sampled_graphs


class NeighborSampler(object):
    def __init__(self, g, num_fanouts):
        self.g = g
        self.num_fanouts = num_fanouts
        
    def sample(self, seeds):
        seeds = torch.LongTensor(seeds)
        blocks = []
        for fanout in reversed(self.num_fanouts):
            if fanout >= self.g.number_of_nodes():
                sampled_graph = dgl.in_subgraph(self.g, seeds)
            else:
                sampled_graph = dgl.sampling.sample_neighbors(self.g, seeds, fanout)
            
            # print(sampled_graph)
            # print(len(seeds))
            sampled_block = dgl.to_block(sampled_graph, seeds, include_dst_in_src=True)
            seeds = sampled_block.srcdata[dgl.NID]
            blocks.insert(0, sampled_block)
            
        return blocks

class SimiLossSampler(object):
    def __init__(self, g):
        self.g = g
        self.metapaths = [
            ('u', 'pos', 'u'), ('u', 'neg', 'u'),
            ('i', 'pos', 'i'), ('i', 'neg', 'i'),
        ]
        self.sub_graphs = {}
        for metapath in self.metapaths:
            self.sub_graphs[metapath] = dgl.transform.metapath_reachable_graph(g, [metapath])

    def sample(self, u_seeds, pos_i_seeds, neg_i_seeds, pos_num=1, neg_num=5):
        u_seeds = torch.LongTensor(u_seeds)
        pos_i_seeds = torch.LongTensor(pos_i_seeds)
        neg_i_seeds = torch.LongTensor(neg_i_seeds)

        user_blocks = []
        pos_item_blocks = []
        neg_item_blocks = []
        for metapath in self.metapaths:
            if metapath[1] == 'pos':
                fanout = pos_num
            else:
                fanout = neg_num
            if metapath[0] == 'u':
                sampled_graph = dgl.sampling.sample_neighbors(self.sub_graphs[metapath], u_seeds, fanout)
                sampled_block = dgl.to_block(sampled_graph, seeds)
                blocks.append(sampled_block)
            else:
                sampled_graph_pos = dgl.sampling.sample_neighbors(self.sub_graphs[metapath], pos_i_seeds, fanout)
                sampled_graph_neg = dgl.sampling.sample_neighbors(self.sub_graphs[metapath], neg_i_seeds, fanout)
                sampled_block_pos = dgl.to_block(sampled_graph_pos, seeds)
                sampled_block_neg = dgl.to_block(sampled_graph_neg, seeds)
                pos_item_blocks.append(sampled_block_pos)
                neg_item_blocks.append(sampled_block_neg)
        # blocks = []
        # for fanout in reversed(self.num_fanouts):
        #     if fanout >= self.g.number_of_nodes():
        #         sampled_graph = dgl.in_subgraph(self.g, seeds)
        #     else:
        #         sampled_graph = dgl.sampling.sample_neighbors(self.g, seeds, fanout)
            
        #     # print(sampled_graph)
        #     # print(len(seeds))
        #     sampled_block = dgl.to_block(sampled_graph, seeds)
        #     seeds = sampled_block.srcdata[dgl.NID]
        #     blocks.insert(0, sampled_block)
            
        return user_blocks, pos_item_blocks, neg_item_blocks


class BipartiteGraphSampler(object):
    def __init__(self, g, sample_num_list):
        self.g = g
        self.sample_num_list = sample_num_list
        self.next_dict = {
            g.ntypes[0]: g.ntypes[1],
            g.ntypes[1]: g.ntypes[0]}

    def sample(self, ntype, seeds):
        seeds = torch.LongTensor(seeds)
        blocks = []
        curr_ntype = ntype
        for sample_num in self.sample_num_list:
            sampled_graph = dgl.sampling.sample_neighbors(self.g, {curr_ntype: seeds}, sample_num)
            # print(sampled_graph)
            sampled_block = dgl.to_block(sampled_graph, {curr_ntype: seeds}, include_dst_in_src=True)
            # print(sampled_block)
            curr_ntype = self.next_dict[curr_ntype]
            seeds = sampled_block.nodes[curr_ntype].data[dgl.NID]
            blocks.append(sampled_block)

        return blocks

