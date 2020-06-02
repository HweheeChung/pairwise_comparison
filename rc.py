import numpy as np
import os
import random
import operator
from scipy.linalg import eig
from scipy.stats import kendalltau as tau
from typing import List, Tuple, Dict, TypeVar
import matplotlib.pyplot as plt
import argparse
import time
from scipy.optimize import minimize

T = TypeVar('T')


def extract_rc_scores(comparisons: List[Tuple[T, T]], regularized: bool = True) -> Dict[T, float]:
    """
    Computes the Rank Centrality scores given a list of pairwise comparisons based on Negahban et al 2016 [1].

    Note it is assumed that the comparisons cannot result in a draw. If you want to include draws, then you can
    treat a draw between `A` and `B` as `A` winning over `B` AND `B` winning over `A`. So for a draw, you can add
    `(A, B)` and `(B, A)` to `comparisons`.

    The regularized version is also implemented. This could be useful when the number of comparisons are small
    with respect to the number of unique items. Note that for properly ranking, number of samples should be in the
    order of n logn, where n is the number of unique items.

    References

    1- Negahban, Sahand et al. “Rank Centrality: Ranking from Pairwise Comparisons.” Operations Research 65 (2017):
    266-287. DOI: https://doi.org/10.1287/opre.2016.1534

    :param comparisons: List of pairs, in `[(winnner, loser)]` format.

    :param regularized: If True, assumes a Beta prior.

    :return: A dictionary of `item -> score`
    """


    winners, losers = zip(*comparisons)
    unique_items = np.hstack([np.unique(winners), np.unique(losers)])
    #unique_items = np.unique(np.hstack([winners, losers]))

    item_to_index = {item: i for i, item in enumerate(unique_items)}

    A = np.ones((len(unique_items), len(unique_items))) * regularized  # Initializing as ones results in the Beta prior

    for w, l in comparisons:
        A[item_to_index[l], item_to_index[w]] += 1

    A_sum = (A[np.triu_indices_from(A, 1)] + A[np.tril_indices_from(A, -1)]) + 1e-6  # to prevent division by zero
    #idx = np.triu_indices_from(A, 1) # value for normalize Aij + Aji = 1
    #A_sum = A[idx] + A[idx[::-1]]

    A[np.triu_indices_from(A, 1)] /= A_sum
    A[np.tril_indices_from(A, -1)] /= A_sum
    #A[idx] /= A_sum
    #A[idx[::-1]] /= A_sum

    d_max = np.max(np.sum(A, axis=1)) # deg is determined after normalize? interesting..
    A /= d_max # no modification on diagonal terms?

    w, v = eig(A, left=True, right=False) # findout pi^T = pi^T A

    max_eigv_i = np.argmax(w)
    scores = np.real(v[:, max_eigv_i])

    return {item: scores[index] for item, index in item_to_index.items()}

def loss(comparisons, lmbd=1e-4):
    # define loss function w.r.t. comparisons result
    # input
    #  coomparisons: list of pairs
    #  lmbd: hyperparameter for l2 reg
    # output
    #  lsf: loss function
    #  gdf: gradient function
    def lsf(x):
        ls = 0
        n = len(comparisons)
        for pair in comparisons:
            wv, lv = x[pair]
            ls += 1/n* np.log(1+np.exp(lv-wv))
        ls += 0.5 * lmbd * np.dot(x,x)
        return ls
    def gdf(x):
        gd = np.zeros_like(x)
        n = len(comparisons)
        for pair in comparisons:
            wv, lv = x[pair]
            xl = np.zeros_like(x)
            xl[pair[1]] = 1; xl[pair[0]] = -1
            gd += 1/n * xl * np.exp(lv-wv)/(1+np.exp(lv-wv))
        gd += lmbd * x
        return gd

    return lsf, gdf


def my_extract_scores(n_players, comparisons: List[Tuple[T, T]], method='rc') -> Dict[T, float]:

    #winners, losers = zip(*comparisons)
    #unique_items = np.unique(np.hstack([winners, losers]))

    #item_to_index = {item: i for i, item in enumerate(unique_items)}

    #A = np.ones((len(unique_items), len(unique_items))) # Initializing as ones results in the Beta prior
    #A = np.zeros((len(unique_items), len(unique_items)))

    A = np.zeros((n_players, n_players))
    #A = np.ones((n_players, n_players))


    if method == 'RC':
        for w, l in comparisons:
            #A[item_to_index[l], item_to_index[w]] += 1
            A[l, w] += 1
        # normalize Aij + Aji = 1
        idx = np.triu_indices_from(A, 1)
        A_sum = A[idx] + A[idx[::-1]]
        # normalize except for zeros (i.e., disconnected node)
        A_sum = A_sum + (A_sum == 0)
        A[idx] /= A_sum
        A[idx[::-1]] /= A_sum

        # transition matrix P
        d_max = np.max(np.sum((A>0), axis=1)) # stronger diagonal element
        #d_max = np.max(np.sum(A, axis=1)) # weaker diagonal element
        P = A/d_max
        # diagonal terms for transition matrix
        P_diag = 1 - np.sum(P, axis=1)
        #assert((P_diag >= 0).all()) # diagonal elements must be positive
        rng = np.arange(len(P))
        P[rng, rng] = P_diag

        ## another method for calculate P
        ## pi(i) = \sum_j{ pi(j) \frac{A_{ji}}{\sum_l {A_{il}}} }
        #A_sum2 = np.sum(A, axis=1)
        #A_sum2 = A_sum2 + (A_sum2 == 0) # mask zeros for x/0 case
        #A_msk = np.tile(A_sum2, (len(A),1))
        #P = A / A_msk

        # find stationary distribution pi
        w, v = eig(P, left=True, right=False) # findout pi^T = pi^T A
        max_eigv_i = np.argmax(w)
        scores = np.real(v[:, max_eigv_i])

    elif method == 'Borda':
        for w, l in comparisons:
            A[l, w] += 1
            A[w, l] -= 1
        scores = np.sum(A, axis=0)

    elif method == 'MLE':
        lsf, gdf = loss(comparisons)
        scores = np.zeros(n_players)
        opt = minimize(lsf, scores, method='BFGS', jac=gdf,
                       options={'disp': False})
                       #options={'disp': True})
        scores = opt.x

    return {i: scores[i] for i in range(n_players)}



def generate_matches(n_trials, n_players, p = 0.9, rep=1, method='Bern'):
    # n_trials: number of trials (matches)
    # n_players: number of players
    # p: probability where higher ranker wins the lower
    # rep: number of repetition to estimate ground truth matching probability

    matches = []
    players = np.arange(n_players)

    for k in range(n_trials//rep):
        # matching policy
        match = np.random.choice(players, size=2, replace=False) # randomly select two players
        match = np.sort(match)
        if method == 'Bern':
            pass # use predefined p
        elif method == 'BTL':
            p = (n_players-match[0]) / (n_players-match[0] + n_players-match[1])
        for j in range(rep):
            rv = np.random.binomial(1, p)
            match = match if rv == 1 else match[::-1]
            matches.append(match)

    return matches

def rank_accuracy(ls, k=None):
    # measure accuracy of ls whether it is sorted or not
    # ls: input list of integer
    # k: integer k used to check accuracy of ls upto k
    if k == None:
        lst = ls
    else:
        lst = ls[0:k]
    ac = [i==l for i,l in enumerate(lst)]
    return np.mean(ac)

def topk_accuracy(ls, k=None):
    # measure accuracy of ls whether it is sorted or not
    # ls: input list of integer
    # k: integer k used to check accuracy of ls upto k
    if k == None:
        lst = ls
    else:
        lst = ls[0:k]
    ac = [0 <= e and e < k for e in lst]
    return np.mean(ac)

# a simple example below
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_players', type=int, default=100)
    parser.add_argument('--sample_method', type=str, default='Bern')
    # sample_method: Bern, BTL
    parser.add_argument('--rank_method', type=str, default='RC')
    # rank_method: RC, Borda, RC_org
    args = parser.parse_args()
    print(args)
    n_players = args.n_players
    sample_method = args.sample_method
    rank_method = args.rank_method
    t1 = time.time()

    fig = plt.figure()
    fig.suptitle('{}'.format(rank_method),y=0.05)
    plt.rc('xtick', labelsize=8)
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,1,2)
    #ax1.set_ylim([0,1])
    ax2.set_ylim([-1,1])
    ax3.set_ylim([0,1])
    ps = [1.0] if sample_method == 'BTL' else [0.95, 0.9, 0.8]
    for p in ps:
        raccs = []; rvaris = []
        taccs = []; tvaris = []
        ktaus = []; kvaris = []
        #n_trials = [1000, 2000, 5000, 7000, 10000, 20000, 50000, 100000]
        #n_trials = [100, 200, 500, 700, 1000, 2000, 5000, 10000, 50000, 100000]
        #n_trials = [100, 200, 500, 700, 1000, 2000, 5000, 10000, 50000]
        n_trials = [1000, 2000, 5000, 10000, 50000, 100000, 150000, 250000]
        #n_trials = [10, 20, 50, 100, 500, 1000, 2000]
        #n_trials = [1000, 10000, 50000]
        for n_trial in n_trials:
            temps = [[],[],[]]
            for i in range(5): # multiple times repeat to minimize randomness
                matches = generate_matches(n_trial, n_players, p, rep=1, method=sample_method)
                if rank_method == 'RC_org':
                    team_to_score = extract_rc_scores(matches) #original code in github
                else:
                    team_to_score = my_extract_scores(n_players, matches, method=rank_method)
                tts = list(team_to_score.items())
                random.shuffle(tts)
                sorted_teams = sorted(tts, key=operator.itemgetter(1), reverse=True)
                sorted_teams_only = [st[0] for st in sorted_teams]
                racc = rank_accuracy(sorted_teams_only)
                tacc = topk_accuracy(sorted_teams_only, k= n_players//10)
                ktau = tau(sorted_teams_only, np.arange(len(sorted_teams_only)))[0]
                temps[0].append(racc)
                temps[1].append(tacc)
                temps[2].append(ktau)

            raccs.append(np.mean(temps[0]))
            taccs.append(np.mean(temps[1]))
            ktaus.append(np.mean(temps[2]))
            rvaris.append(np.sqrt(np.var(temps[0])))
            tvaris.append(np.sqrt(np.var(temps[1])))
            kvaris.append(np.sqrt(np.var(temps[2])))
        raccs = np.array(raccs)
        taccs = np.array(taccs)
        ktaus = np.array(ktaus)
        rvaris = np.array(rvaris)
        tvaris = np.array(tvaris)
        kvaris = np.array(kvaris)
        n_trials_str = [str(e) for e in n_trials]
        ax1.plot(n_trials_str, raccs, label=str(p))
        ax1.fill_between(n_trials_str, raccs-rvaris, raccs+rvaris, alpha=0.3)
        ax3.plot(n_trials_str, taccs, label=str(p))
        ax3.fill_between(n_trials_str, taccs-tvaris, taccs+tvaris, alpha=0.3)
        ax2.plot(n_trials_str, ktaus, label=str(p))
        ax2.fill_between(n_trials_str, ktaus-kvaris, ktaus+kvaris, alpha=0.3)
        # change this to mx and mn instead of sigma?
        print(args, ':{}: {} elapsed'.format(p, time.time()-t1))

    ax1.legend()
    ax2.legend()
    ax3.legend()
    #ax1.title.set_text('RankACC')
    #ax2.title.set_text('TopACC')
    #ax3.title.set_text('KDTau')
    ax1.set_title('RankACC')
    ax3.set_title('TopACC')
    ax2.set_title('KDTau')
    fn = '{}players_{}_{}.png'.format(n_players, sample_method, rank_method)
    if not os.path.exists('figures'):
        os.mkdir('figures')
    save_dir = os.path.join('figures', fn)
    fig.tight_layout(h_pad=1, w_pad=0)
    plt.savefig(save_dir)
    #plt.show()

    #matches = generate_matches(1000, 10, 0.8)
    #team_to_score = my_extract_rc_scores(matches)
    #sorted_teams = sorted(team_to_score.items(), key=operator.itemgetter(1), reverse=True)
    #for team, score in sorted_teams:
    #    print('{} has a score of {!s}'.format(team, round(score,3)))
    #sorted_teams_only = [st[0] for st in sorted_teams]
    #acc = rank_accuracy(sorted_teams_only)
    #print(acc)

    print('done')
