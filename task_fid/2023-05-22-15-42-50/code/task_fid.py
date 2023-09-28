import torch as t
import torch.nn as nn
import torch.backends.cudnn as cudnn
from config import cifar10_config
from utils import *
from nets import weights_init
from dataset import get_dataset
from torch.autograd import Variable
import numpy as np
import random
import itertools
import datetime
import argparse
from argparse import Namespace
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
mse = nn.MSELoss(reduction='sum').cuda()

def langevin_x(x, netE, args, should_print=True):
    l_steps = args.e_l_steps
    l_step_size = args.e_l_step_size
    n_step_size = args.e_n_step_size
    x = x.clone().detach()
    x.requires_grad = True
    for istep in range(l_steps):
        energy = netE(x).sum()
        # errRecon = mse(inputV_recon, x) / batch_size
        grad = t.autograd.grad(energy, x)[0]
        x.data = x.data - 0.5 * l_step_size * l_step_size * grad
        if args.use_noise:
            x.data = x.data + n_step_size * t.randn_like(x).data

        if (istep % 1 == 0 or istep == l_steps - 1) and should_print:
            logging.info('Langevin X {}/{}: energy: {}'.format(istep + 1, l_steps, energy.item()))
    return x.detach()

def sample_x(batch_size, netG, netE, args):
    noise = t.randn(batch_size, args.nz, 1, 1).to(args.device)
    noiseV = Variable(noise)
    samples = netG(noiseV)
    samples_corr = langevin_x(samples, netE, args, should_print=False)
    return samples, samples_corr

def compute_fid(netG, netE, netI, args):

    to_range_0_1 = lambda x: (x + 1.) / 2. if args.normalize_data else x

    from pytorch_fid_jcui7.fid_score import compute_fid
    from tqdm import tqdm
    try:
        s1 = []
        s2 = []
        for _ in tqdm(range(int(50000 / args.batch_size))):
            noise = t.randn(args.batch_size, args.nz, 1, 1).to(args.device)
            noiseV = Variable(noise)
            syn = netG(noiseV).detach()
            syn_corr = langevin_x(syn, netE, args, should_print=False)

            syn = to_range_0_1(syn).clamp(min=0., max=1.)
            s1.append(syn)
            syn_corr = to_range_0_1(syn_corr).clamp(min=0., max=1.)
            s2.append(syn_corr)

        s1 = t.cat(s1)
        fid1 = compute_fid(x_train=None, x_samples=s1, path=args.fid_stat_dir)
        s2 = t.cat(s2)
        fid2 = compute_fid(x_train=None, x_samples=s2, path=args.fid_stat_dir)
        print(f'fid gen: {fid1:.5f} fid ebm: {fid2:.5f}')
        show_single_batch(s1[:100], args.dir + f'gen_{fid1:.4f}.png', nrow=10)
        show_single_batch(s2[:100], args.dir + f'ebm_{fid2:.4f}.png', nrow=10)
        return

    except Exception as e:
        print(e)

def build_netG(args):
    from nets import _Cifar10_netG as _netG
    netG = _netG(nz=args.nz, ngf=args.ngf)
    netG.apply(weights_init)
    netG.to(args.device)
    return netG

def build_netE(args):
    from nets import _Cifar10_netE as _netE
    netE = _netE(nc=3, ndf=args.ndf)
    netE.apply(weights_init)
    netE.to(args.device)
    netE = add_sn(netE)
    return netE

def build_netI(args):
    from nets import _Cifar10_netI as _netI
    netI = _netI(nz=args.nz, nif=args.nif)
    netI.apply(weights_init)
    netI.to(args.device)
    return netI

def letgo(args_job, output_dir):
    set_seeds(1234)
    args = parse_args()
    args = overwrite_opt(args, args_job)
    args = overwrite_opt(args, cifar10_config)

    # date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # output_dir += f"/{args.exp_path.split('/')[2]}/{args.exp_ckpt}/{date}/"
    # os.makedirs(output_dir, exist_ok=True)
    output_dir += '/'
    args.dir = output_dir

    loaded_config = load_args(args.exp_path)
    loaded_config = Namespace(**loaded_config)
    print('Loaded Config')
    print(loaded_config)
    loaded_config.device = args.device

    if args.default_gen_ld:
        print('syn default gen ld config')
        args.l_steps = loaded_config.l_steps
        args.l_step_size = loaded_config.l_step_size
        args.n_step_size = loaded_config.n_step_size
        args.prior_sig = loaded_config.prior_sig
        args.sig = loaded_config.sig

    if args.default_ebm_ld:
        print('syn default ebm ld config')
        args.e_l_steps = loaded_config.e_l_steps
        args.e_l_step_size = loaded_config.e_l_step_size
        args.e_n_step_size = loaded_config.e_n_step_size
        args.use_noise = loaded_config.use_noise

    args.nz = loaded_config.nz

    save_args(vars(args), output_dir)

    netG = build_netG(loaded_config)
    netE = build_netE(loaded_config)
    netI = build_netI(loaded_config)

    # ckpt = t.load(args.exp_path + 'ckpt/checkpoint.pth', map_location='cpu')
    ckpt = t.load(args.exp_path + 'ckpt/8.9682_280.pth', map_location='cpu')
    netG.load_state_dict(ckpt['netG'], strict=True)
    netE.load_state_dict(ckpt['netE'], strict=True)
    netI.load_state_dict(ckpt['netI'], strict=True)

    compute_fid(netG, netE, netI, args)
    return

def parse_args():
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument('--exp_path', type=str, default='/Tian-ds/jcui7/triangle-abp/a100_double_MCMC4_t1/2023-04-20-22-39-01/0/')
    parser.add_argument('--batch_size', type=int, default=100)

    parser.add_argument('--default_ebm_ld', type=bool, default=False)
    parser.add_argument('--e_l_steps', type=int, default=70)
    parser.add_argument('--e_l_step_size', type=float, default=0.4)
    parser.add_argument('--e_n_step_size', type=float, default=1e-3)
    parser.add_argument("--use_noise", type=bool, default=True)

    parser.add_argument('--default_gen_ld', type=bool, default=True)
    parser.add_argument('--l_steps', type=int, default=10)
    parser.add_argument('--l_step_size', type=float, default=0.1)
    parser.add_argument('--n_step_size', type=float, default=0.1)
    parser.add_argument('--prior_sig', type=float, default=1.0)
    parser.add_argument('--sig', type=float, default=0.3)

    parser.add_argument('--job_id', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    # Parser
    args, unknown = parser.parse_known_args()
    if len(unknown) > 0:
        print("Invalid arguments %s" % unknown)
        parser.print_help()
        sys.exit()
    return args

def set_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def main():
    output_dir = './{}/'.format(os.path.splitext(os.path.basename(__file__))[0])
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    output_dir += t + '/'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + 'code/', exist_ok=True)

    [save_file(output_dir, f) for f in
    ['config.py', 'dataset.py', 'nets.py', 'utils.py', os.path.basename(__file__)]]

    opt = dict()
    letgo(opt, output_dir)

if __name__ == '__main__':
    main()
