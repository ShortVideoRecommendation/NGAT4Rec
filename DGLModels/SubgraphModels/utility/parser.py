import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run HETERO.")
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')

    parser.add_argument('--data_path', nargs='?', default='../../Data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')

    parser.add_argument('--dataset', nargs='?', default='gowalla',
                        help='Choose a dataset from {gowalla, yelp2018, amazon-book, movielens-20m}')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--test_att', type=int, default=0,
                        help='0: normal, 1: test')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of epoch.')
    parser.add_argument('--multi_loss', type=int, default=0,
                        help='1: use multi-loss')
    parser.add_argument('--mc', type=int, default=0,
                        help='1: use multi-center loss')
    # parser.add_argument('--am', type=int, default=0,
    #                     help='1: use adaptive margin')
    # parser.add_argument('--bm', type=int, default=0,
    #                     help='1: use adaptive margin b')
    parser.add_argument('--use_attribute', type=int, default=0,
                        help='1: use item-side attribute')

    parser.add_argument('--loss_type', type=str, default='bpr',
                        help='bpr, a, b, c')

    parser.add_argument('--opt_level', type=str, default='O1',
                        help='O0, O1, O2, O3')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='attention head.')
    parser.add_argument('--layer_size', nargs='?', default='[64]',
                        help='Output sizes of every layer')
    parser.add_argument('--neighbors_num', nargs='?', default='[100]',
                        help='Sampled neighbors num of every layer')

    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')

    parser.add_argument('--regs', nargs='?', default='[1e-4]',
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--lbd', type=float, default=0.5,
                        help='item center weight')

    parser.add_argument('--pretrain_lr', type=float, default=0.0001,
                        help='Learning rate.')

    parser.add_argument('--model_type', nargs='?', default='hetero',
                        help='Specify the name of model (hetero).')
    parser.add_argument('--adj_type', nargs='?', default='norm',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
    parser.add_argument('--alg_type', nargs='?', default='rw',
                        help='Specify the type of the graph convolutional layer from {rw, rw_single, rw_fixed, rw_single_svd, rw_svd, rw_final, rw_linear}.')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--mess_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[5, 10, 15, 20]',
                        help='Output sizes of every layer')

    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--sample_test_flag', nargs='?', default='full',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')


    return parser.parse_args()
