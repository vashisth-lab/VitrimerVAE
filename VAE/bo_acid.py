import argparse, torch, pickle, sys, os
sys.path.append('bo')
import numpy as np
import pandas as pd
from sparse_gp import SparseGP
from vae import *


def to_numpy(tensors):
    convert = lambda x : x.numpy() if type(x) is torch.Tensor else x
    a,b,c = tensors
    b = [convert(x) for x in b[0]], [convert(x) for x in b[1]]
    return a, b, c


def tensorize(acid, epoxide, vocab_aci, vocab_epo):
    try:
        x_aci = MolGraph.tensorize(acid, vocab_aci, common_atom_vocab)
        x_epo = MolGraph.tensorize(epoxide, vocab_epo, common_atom_vocab)
        return to_numpy(x_aci), to_numpy(x_epo)
    except:
        return None


def predict(z, model, vocab_aci, vocab_epo):
    z_tensor = torch.tensor(z).float().cuda()
    try:
        acid, epoxide = model.decode(z_tensor)
        if not acid[0] or not epoxide[0]:
            return None, None, None
    except:
        return None, None, None
    tensors = tensorize(acid, epoxide, vocab_aci, vocab_epo)
    if not tensors:
        return None, None, None
    z_recon, _ = model.encode(tensors)
    tg_recon = model.predict(z_recon).squeeze().detach().cpu().numpy()
    return acid[0], epoxide[0], tg_recon


def main(args, seed):
    torch.manual_seed(seed)

    vocab = [x.strip("\r\n ").split() for x in open('data/vocab_acid.txt')] 
    args.vocab_aci = PairVocab(vocab)
    vocab = [x.strip("\r\n ").split() for x in open('data/vocab_epoxide.txt')] 
    args.vocab_epo = PairVocab(vocab)

    model = HierVAE(args).cuda()
    model.load_state_dict(torch.load('%s/ckpt/prop49.model' % (args.savedir))[0])
    model.eval()

    with open('data/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('%s/pca.pkl' % (args.savedir), 'rb') as f:
        pca = pickle.load(f)

    df = pd.read_csv('%s/bo_initial_acid_%d.csv' % (args.savedir, seed))
    z = df.loc[:, 'z0':'z'+str(args.latent_size-1)].to_numpy()
    tg_norm = df['tg_norm'].to_numpy()
    z1 = z[:, :(args.latent_size - args.epoxide_size)]
    z2 = z[:, (args.latent_size - args.epoxide_size):(args.latent_size - args.epoxide_size + args.share_size)]
    z3 = z[:, (args.latent_size - args.epoxide_size + args.share_size):]

    X = z1
    target = args.target
    y = np.square(tg_norm.reshape(-1, 1) - scaler.transform(np.array(target).reshape(-1, 1)).squeeze().item())

    np.random.seed(seed)
    n = X.shape[0]
    permutation = np.random.choice(n, n, replace = False)
    X_train = X[permutation, :][:int(n*0.9), :]
    y_train = y[permutation, :][:int(n*0.9), :]
    X_test = X[permutation, :][int(n*0.9):, :]
    y_test = y[permutation, :][int(n*0.9):, :]
    
    vitrimer = []
    tg = []
    pca1 = []
    pca2 = []
    it = []
    iteration = 0
    while iteration < 50:
        print('it ' + str(iteration))
        sys.stdout.flush()
        
        np.random.seed(iteration * seed)
        M = 100
        n_sample = 50
        sgp = SparseGP(X_train, 0 * X_train, y_train, M)
        sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0, y_test, minibatch_size = 10 * M, max_iterations = 100, learning_rate = 0.001)

        z1_next = sgp.batched_greedy_ei(n_sample, np.min(X_train, 0), np.max(X_train, 0))
        z2_next = z2[:n_sample, :]
        z3_next = z3[:n_sample, :]
        z_next = np.concatenate((z1_next, z2_next, z3_next), axis=1)
        y_new = []
        z1_new = []
        for i in range(n_sample):
            z = z_next[i].reshape(1, -1)
            acid, epoxide, tg_norm = predict(z, model, args.vocab_aci, args.vocab_epo)
            if acid and epoxide and tg_norm:
                if check_acid(acid) and check_epoxide(epoxide) and (acid, epoxide) not in vitrimer:
                    it.append(iteration)
                    z1_new.append(z1_next[i].reshape(1, -1))
                    vitrimer.append((acid, epoxide))
                    tg.append(scaler.inverse_transform(np.array(tg_norm).reshape(-1, 1)).squeeze().item())
                    z_pca = pca.transform(z)
                    pca1.append(z_pca[0][0])
                    pca2.append(z_pca[0][1])
                    y_new.append((tg_norm - scaler.transform(np.array(target).reshape(-1, 1)).squeeze().item())**2)

        if len(z1_new) > 0:
            z1_new = np.vstack(z1_new)
            X_train = np.concatenate([X_train, z1_new], 0)
            y_train = np.concatenate([y_train, np.array(y_new)[:, None]], 0)
        iteration += 1

    acid, epoxide = zip(*vitrimer)
    acid = list(acid)
    epoxide = list(epoxide)
    df = pd.DataFrame({'iteration': it, 'acid': acid, 'epoxide': epoxide, 'tg_pred': tg, 
                       'pca1': pca1, 'pca2': pca2})
    df.to_csv('%s/bo_acid%d_%d.csv' % (args.savedir, target, seed), index=False, float_format='%.4f')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', type=str, required=True)
    parser.add_argument('--target', type=int, required=True)
    parser.add_argument('--atom_vocab', default=common_atom_vocab)

    parser.add_argument('--rnn_type', type=str, default='LSTM')
    parser.add_argument('--hidden_size', type=int, default=250)
    parser.add_argument('--embed_size', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--latent_size', type=int, default=128)
    parser.add_argument('--acid_size', type=int, default=112)
    parser.add_argument('--epoxide_size', type=int, default=112)
    parser.add_argument('--share_size', type=int, default=96)
    parser.add_argument('--depthT', type=int, default=15)
    parser.add_argument('--depthG', type=int, default=15)
    parser.add_argument('--diterT', type=int, default=1)
    parser.add_argument('--diterG', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--prop_hidden_size', type=int, default=64)

    args = parser.parse_args()

    for i in range(1, 11):
        if not os.path.isfile('%s/bo_acid%d_%d.csv' % (args.savedir, args.target, i)):
            main(args, i)

