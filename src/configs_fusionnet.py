
import argparse

# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--npy_dir',  default='../data/fusion/', required=False, help='facades')
parser.add_argument('--data_dir',  default='../data/fusion/', required=False, help='facades')
parser.add_argument('--model_name', default='fusion', type=str, help='dataset')

parser.add_argument('--save_dir',  default='../results/fusion/', required=False, help='facades')
parser.add_argument('--dataset',  default='cifar10', required=False, help='facades')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=8, help='8- 32/ 64-256generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=8, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=300, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=300, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='coefficient for weight decay')
parser.add_argument('--drop_rate', default=0.1,
                    type=float, help='dropout rate')
parser.add_argument('-drop_two', '--drop_two', dest='drop_two',
                    action='store_true', help='2d dropout on')
parser.add_argument('-nesterov', '--nesterov', dest='nesterov', action='store_true',
                    help='nesterov momentum')

parser.add_argument('--save_steps', type=int, default=10)
parser.add_argument('--log_steps', type=int, default=30)

parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
parser.add_argument('--flag_retrain', action='store_true', help='resume')
parser.add_argument('--concat_input', action='store_true', help='resume')

parser.add_argument('--nactors', type=int, default=2, help='n-servers')

args = parser.parse_args()
