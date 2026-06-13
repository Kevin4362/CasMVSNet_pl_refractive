import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/mvs_training/dtu/',
                        help='root directory of dtu dataset')
    parser.add_argument('--dataset_name', type=str, default='dtu',
                        choices=['dtu', 'blendedmvs'],
                        help='which dataset to train/val')
    parser.add_argument('--n_views', type=int, default=3,
                        help='number of views (including ref) to be used in training')
    parser.add_argument('--levels', type=int, default=3, choices=[3],
                        help='number of FPN levels (fixed to be 3!)')
    parser.add_argument('--depth_interval', type=float, default=2.65,
                        help='depth interval for the finest level, unit in mm')
    parser.add_argument('--uw_degradations', nargs='+', type=str, default=['Bluish'],
                        choices=['Bluish', 'Greenish', 'Hazy', 'Lowlight', 'all'],
                        help='underwater image degradation types to use; use all for four types')
    parser.add_argument('--scan_list_dir', type=str, default=None,
                        help='directory containing train.txt/val.txt/test.txt; default uses root_dir if present')
    parser.add_argument('--n_depths', nargs='+', type=int, default=[8,32,48],
                        help='number of depths in each level')
    parser.add_argument('--interval_ratios', nargs='+', type=float, default=[1.0,2.0,4.0],
                        help='depth interval ratio to multiply with --depth_interval in each level')
    parser.add_argument('--num_groups', type=int, default=1, choices=[1, 2, 4, 8],
                        help='number of groups in groupwise correlation, must be a divisor of 8')
    parser.add_argument('--loss_type', type=str, default='sl1',
                        choices=['sl1'],
                        help='loss to use')
    parser.add_argument('--use_refractive', default=False, action='store_true',
                        help='use differentiable flat-port refractive camera warping')
    parser.add_argument('--z_inner', type=float, default=10.0,
                        help='distance from camera center to inner glass surface')
    parser.add_argument('--glass_thickness', type=float, default=5.0,
                        help='flat-port glass thickness')
    parser.add_argument('--n_air', type=float, default=1.0,
                        help='air refractive index')
    parser.add_argument('--n_glass', type=float, default=1.52,
                        help='glass refractive index')
    parser.add_argument('--n_water', type=float, default=1.333,
                        help='water refractive index')
    parser.add_argument('--refractive_newton_iters', type=int, default=4,
                        help='Newton iterations for differentiable refractive projection')
    parser.add_argument('--refractive_depth_chunk', type=int, default=4,
                        help='number of depth hypotheses projected at once in refractive warp')
    parser.add_argument('--freeze_refractive_params', default=False, action='store_true',
                        help='keep z_inner, glass_thickness and n_glass fixed')
    parser.add_argument('--cost_reg_checkpoint', default=False, action='store_true',
                        help='checkpoint 3D cost regularization to reduce training memory')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--num_epochs', type=int, default=16,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')

    parser.add_argument('--ckpt_path', type=str, default='',
                        help='pretrained checkpoint path to load')
    parser.add_argument('--prefixes_to_ignore', nargs='+', type=str, default=['loss'],
                        help='the prefixes to ignore in the checkpoint state dict')

    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer type',
                        choices=['sgd', 'adam', 'radam', 'ranger'])
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='steplr',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    #### params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')
    ###########################
    #### params for steplr ####
    parser.add_argument('--decay_step', nargs='+', type=int, default=[20],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.1,
                        help='learning rate decay amount')
    ###########################
    #### params for poly ####
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='exponent for polynomial learning rate decay')
    ###########################

    parser.add_argument('--use_amp', default=False, action="store_true",
                        help='use mixed precision training (NOT SUPPORTED!)')

    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')

    return parser.parse_args()
