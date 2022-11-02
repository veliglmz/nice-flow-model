import argparse


def arg_parse():
    parser = argparse.ArgumentParser(
        description = '''A minimal implementation of Laurent Dinh David Krueger Yoshua Bengio
                        "NICE: NON-LINEAR INDEPENDENT COMPONENTS ESTIMATION" (2015)''',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--n_samples',
                        type = int, default = 10000,
                        help = 'Number of training sample size')

    parser.add_argument('--n_epochs',
                        type = int, default = 10000,
                        help = 'How many epochs')

    parser.add_argument('--batch_size',
                        type = int, default = 512,
                        help = 'Batch size')

    parser.add_argument('--lr',
                        type = float, default = 1e-3,
                        help = 'Learning rate')
    
    parser.add_argument('--n_inputs',
                        type = int, default = 2,
                        help = 'Input dimension size (only implemented 2D)')
    
    parser.add_argument('--n_layers',
                        type = int, default = 5,
                        help = 'Number of hidden layers for transformation')

    parser.add_argument('--n_hiddens',
                        type = int, default = 64,
                        help = 'Number of hidden neurons for transformation')
    
    parser.add_argument('--beta1',
                        type = float, default = 0.9,
                        help = 'Value of Beta 1 for Adam optimizer')

    parser.add_argument('--beta2',
                        type = float, default = 0.999,
                        help = 'Value of Beta 2 for Adam optimizer')

    parser.add_argument('--eps',
                        type = float, default = 1e-4,
                        help = 'Value of epsilon for Adam optimizer')
    
    parser.add_argument('--model_path',
                        type = str, default = "nice.pth",
                        help = 'Model path to save')
    args = parser.parse_args()
    return args
