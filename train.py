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

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
mse = nn.MSELoss(reduction='sum').cuda()

def reparametrize(mu, log_sigma, is_train=True):
    if is_train:
        std = t.exp(log_sigma.mul(0.5))
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    else:
        return mu

def diag_normal_NLL(z, z_mu, z_log_sigma):
    # define the Negative Log Probability of Normal which has diagonal cov
    # input: [batch nz, 1, 1] squeeze it to batch nz
    # return: shape is [batch]
    nll = 0.5 * t.sum(z_log_sigma.squeeze(), dim=1) + \
          0.5 * t.sum((t.mul(z - z_mu, z - z_mu) / (1e-6 + t.exp(z_log_sigma))).squeeze(), dim=1)
    return nll.squeeze()

def langevin_z(z, x, netG, args, should_print=True):
    l_steps = args.l_steps
    l_step_size = args.l_step_size
    n_step_size = args.n_step_size
    prior_sig = args.prior_sig
    batch_size = x.shape[0]
    z = z.clone().detach()
    z.requires_grad = True
    for istep in range(l_steps):
        inputV_recon = netG(z)
        errRecon = mse(inputV_recon, x)
        # errRecon = mse(inputV_recon, x) / batch_size
        grad = t.autograd.grad(errRecon, z)[0]

        z.data = z.data - 0.5 * l_step_size * l_step_size * (grad / args.sig / args.sig + 1.0 / (
                prior_sig * prior_sig) * z.data) + n_step_size * t.randn_like(z).data

        if (istep % 1 == 0 or istep == l_steps - 1) and should_print:
            logging.info('Langevin Z {}/{}: MSE: {}'.format(istep + 1, l_steps, errRecon.item()))
    return z.detach()

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

def recon_x(x, netG, netI, args):
    infer_z_mu_input, _ = netI(x)
    recon_input = netG(infer_z_mu_input).detach()
    infer_z_mu_input_corrected = langevin_z(infer_z_mu_input, x, netG, args, should_print=False)
    recon_input_corr = netG(infer_z_mu_input_corrected).detach()
    return recon_input, recon_input_corr

def fit(netG, netE, netI, dl_train, test_batch, args, logger):

    optG = t.optim.Adam(netG.parameters(), lr=args.lrG, weight_decay=args.g_decay, betas=(args.beta1G, 0.9))
    optE = t.optim.Adam(netE.parameters(), lr=args.lrE, weight_decay=args.e_decay, betas=(args.beta1E, 0.9))
    optI = t.optim.Adam(netI.parameters(), lr=args.lrI, weight_decay=args.i_decay, betas=(args.beta1I, 0.9))
    lrG_schedule = t.optim.lr_scheduler.ExponentialLR(optG, args.g_gamma)
    lrE_schedule = t.optim.lr_scheduler.ExponentialLR(optE, args.e_gamma)
    lrI_schedule = t.optim.lr_scheduler.ExponentialLR(optI, args.i_gamma)

    import math
    fid_best_syn = math.inf
    fid_best_ep = 0
    to_range_0_1 = lambda x: (x + 1.) / 2. if args.normalize_data else x
    log_iter = int(len(dl_train) // 4) if len(dl_train) > 1000 else int(len(dl_train) // 2)

    noise = t.randn(args.batch_size, args.nz, 1, 1).to(args.device)

    for ep in range(args.epochs):
        lrE_schedule.step(epoch=ep)
        lrG_schedule.step(epoch=ep)
        lrI_schedule.step(epoch=ep)

        for i, x in enumerate(dl_train, 0):
            if i % log_iter == 0:
                logger.info(
                    "==" * 10 + f"ep: {ep} batch: [{i}/{len(dl_train)}] best_fid_syn: {fid_best_syn:.3f} best_fid_ep: {fid_best_ep}" + "==" * 10)

            training_log = f"[{ep}/{args.epochs}][{i}/{len(dl_train)}] best_fid_syn: {fid_best_syn:.3f} fid best ep: {fid_best_ep} \n"

            x = x[0].to(args.device) if type(x) is list else x.to(args.device)
            batch_size = x.shape[0]

            """
            Train G: use the inferred z
            """
            optG.zero_grad()
            noise.resize_(batch_size, args.nz, 1, 1).normal_()
            noiseV = Variable(noise)
            samples = netG(noiseV)

            infer_z_mu_true, infer_z_log_sigma_true = netI(x)
            z_input = reparametrize(infer_z_mu_true, infer_z_log_sigma_true)
            z_input_corr = langevin_z(z_input, x, netG, args, should_print=(i % log_iter == 0))

            samples_corr = langevin_x(samples, netE, args, should_print=(i % log_iter == 0))
            inputV_recon = netG(z_input_corr.detach())
            errRecon = mse(inputV_recon, x) / batch_size
            errSample = mse(samples, samples_corr) / batch_size

            errG = errRecon + args.Sfactor * errSample
            errG.backward()
            optG.step()
            with t.no_grad():
                et = netE(x).squeeze().mean()
            training_log += f"{'errG':<20}: {errG.item():<20.2f} {'errRecon':<20}: {errRecon.item():<20.2f} {'errSample':<20}: {errSample.item():<20.2f}  {'E_T':<20}: {et.item():<20.2f}\n"

            """
            Train I: use the inferred z
            """
            optI.zero_grad()
            neg_log_q_z = t.mean(diag_normal_NLL(z_input_corr, infer_z_mu_true, infer_z_log_sigma_true))
            infer_z_mu_gen, infer_z_log_sigma_gen = netI(samples_corr)
            errLatent = t.mean(diag_normal_NLL(noiseV, infer_z_mu_gen, infer_z_log_sigma_gen))
            errI = neg_log_q_z + args.Ifactor * errLatent
            errI.backward()
            optI.step()
            training_log += f"{'errI':<20}: {errI.item():<20.2f} {'neg_log_q_z':<20}: {neg_log_q_z.item():<20.2f} {'errLatent':<20}: {errLatent.item():<20.2f}\n"

            """
            Train E
            """
            optE.zero_grad()
            Eng_T = netE(x).squeeze()
            E_T = t.mean(Eng_T)
            samples = netG(noiseV) # G should catch up within per-iteration, quicker, fine_tune to get better.
            if args.fine_tune:
                samples = langevin_x(samples, netE, args, should_print=False)
            Eng_F = netE(samples.detach()).squeeze()
            E_F = t.mean(Eng_F)
            errE = (E_T - E_F)
            errE_loss = errE / (args.e_n_step_size/args.e_l_step_size)**2
            errE_loss.backward()
            optE.step()
            training_log += f"{'errE':<20}: {errE_loss.item():<20.2f} {'E_T':<20}: {E_T.item():<20.2f} {'E_F':<20}: {E_F.item():<20.2f}\n"

            if i % log_iter == 0:
                logger.info(training_log)

            if errE > 1e4 or errE < -1e4:
                logger.info("explode at ep {} iter {}".format(ep, i))
                logger.info(training_log)
                return

            if t.isnan(errE) or t.isnan(errG) or t.isnan(errI):
                logger.info("explode at ep {} iter {}".format(ep, i))
                logger.info(training_log)
                return

        if ep % args.vis_iter == 0:
            rec_imgs_dir = args.dir + 'imgs/rec/'
            os.makedirs(rec_imgs_dir, exist_ok=True)
            syn_imgs_dir = args.dir + 'imgs/syn/'
            os.makedirs(syn_imgs_dir, exist_ok=True)

            recon_input, recon_input_corr = recon_x(test_batch, netG, netI, args)
            show_single_batch(recon_input, rec_imgs_dir + f'{ep:>07d}_rec.png', nrow=10)
            show_single_batch(recon_input_corr, rec_imgs_dir + f'{ep:>07d}_rec_corr.png', nrow=10)

            samples, samples_corr = sample_x(args.batch_size, netG, netE, args)
            show_single_batch(samples, syn_imgs_dir + f'{ep:>07d}_syn_gen.png', nrow=10)
            show_single_batch(samples_corr, syn_imgs_dir + f'{ep:>07d}_syn_corr.png', nrow=10)

            os.makedirs(args.dir + '/ckpt', exist_ok=True)
            save_dict = {
                'epoch': ep,
                'netG': netG.state_dict(),
                'netE': netE.state_dict(),
                'netI': netI.state_dict(),
                'optG': optG.state_dict(),
                'optE': optE.state_dict(),
                'optI': optI.state_dict(),
            }
            t.save(save_dict, '{}/{}.pth'.format(args.dir + '/ckpt', ep))
            keep_last_ckpt(path=args.dir + '/ckpt/', num=10)

    return

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
    output_dir += '/'
    args.dir = output_dir

    [os.makedirs(args.dir + f'{f}/', exist_ok=True) for f in ['ckpt', 'imgs']]

    logger = Logger(args.dir, f"job{args.job_id}")
    logger.info('Config')
    logger.info(args)

    save_args(vars(args), output_dir)

    ds_train, ds_val, input_shape = get_dataset(args)
    dl_train = t.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    dl_val = t.utils.data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    args.input_shape = input_shape
    logger.info("Training samples %d" % len(ds_train))

    fix_x = next(iter(dl_train))
    test_batch = fix_x[0].to(args.device) if type(fix_x) is list else fix_x.to(args.device)
    show_single_batch(test_batch, args.dir + 'imgs/test_batch.png', nrow=int(args.batch_size ** 0.5))

    netG = build_netG(args)
    netE = build_netE(args)
    netI = build_netI(args)

    logger.info(f"netG params: {compute_model_params(netG)}")
    logger.info(f"netE params: {compute_model_params(netE)}")
    logger.info(f"netI params: {compute_model_params(netI)}")

    fit(netG, netE, netI, dl_train, test_batch, args, logger)

    return

def parse_args():
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument('--epochs', type=int, default=201)
    parser.add_argument('--batch_size', type=int, default=100)

    parser.add_argument('--lrE', type=float, default=1e-4, help='learning rate for E, default=0.0002')
    parser.add_argument('--lrG', type=float, default=3e-4, help='learning rate for GI, default=0.0002')
    parser.add_argument('--lrI', type=float, default=1e-4, help='learning rate for GI, default=0.0002')

    parser.add_argument('--Sfactor', type=float, default=50.0)
    parser.add_argument('--Ifactor', type=float, default=0.1)

    parser.add_argument('--e_l_steps', type=int, default=30)
    parser.add_argument('--e_l_step_size', type=float, default=0.5)
    parser.add_argument('--e_n_step_size', type=float, default=0.001)
    parser.add_argument("--use_noise", type=bool, default=True)
    parser.add_argument("--fine_tune", type=bool, default=False)

    parser.add_argument('--ngf', type=int,  default=512)
    parser.add_argument('--ndf', type=int,  default=512)
    parser.add_argument('--nif', type=int,  default=128)

    parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector') # 100
    parser.add_argument('--sig', type=float, default=0.3, help='size of the latent z vector') # 100

    parser.add_argument('--l_steps', type=int, default=10)
    parser.add_argument('--l_step_size', type=float, default=0.1)
    parser.add_argument('--n_step_size', type=float, default=0.1)
    parser.add_argument('--prior_sig', type=float, default=1.0)

    parser.add_argument('--beta1E',  type=float, default=0., help='beta1 for adam. default=0.5')
    parser.add_argument('--beta1G',  type=float, default=0., help='beta1 for adam GI. default=0.5')
    parser.add_argument('--beta1I',  type=float, default=0., help='beta1 for adam GI. default=0.5')
    parser.add_argument('--e_decay', type=float, default=0.0000, help='weight decay for E')
    parser.add_argument('--i_decay', type=float, default=0.0005, help='weight decay for I')
    parser.add_argument('--g_decay', type=float, default=0.0005, help='weight decay for G')
    parser.add_argument('--e_gamma', type=float, default=0.998, help='lr decay for EBM')
    parser.add_argument('--i_gamma', type=float, default=0.998, help='lr decay for I')
    parser.add_argument('--g_gamma', type=float, default=0.998, help='lr decay for G')

    parser.add_argument('--vis_iter', type=int, default=1)

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
