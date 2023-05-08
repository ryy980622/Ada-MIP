import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
    parser.add_argument('--DS', dest='DS', default='PROTEINS', help='Dataset')
    parser.add_argument('--local', dest='local', action='store_const', 
            const=True, default=True)
    parser.add_argument('--glob', dest='gloWb', action='store_const',
            const=True, default=True)
    parser.add_argument('--prior', dest='prior', action='store_const', 
            const=True, default=False)

    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.', default=0.001)
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=3,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=64,
            help='')

    parser.add_argument('--aug', type=str, default='all')
    #parser.add_argument('--aug', type=str, default='dnodes')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--num_view', type=int, default=4)
    parser.add_argument('--kernels', type=list, default=['WL', 'PM', 'WLOA'])
    parser.add_argument('--kernel', type=str, default='WL')
    parser.add_argument('--device', type=str, default='cuda:3')
    return parser.parse_args()
'''
IMDB-B(256,10*all,0.05,0.05, 64,LR:0.0005,init_dim:128), IMDB-M(10*all, 128, LR:0.0001,init_dim:96, 64)
PTC(512,10*all,0.005,64), MUTAG(128,all,0.005,0.005,64,0.001), PROTEINS(128), NCI1(128,1*all), DD(128, 10*all,0.005): lr 0.001
'''
