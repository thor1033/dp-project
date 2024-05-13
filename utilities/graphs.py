
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.tr_model import solve

def agg_holdings(mp, s, sol):
    """
    Syntax: graphs.agg_holdings(mp, s, sol)
    DISTRIBUTION PLOT: ALL ON ONE AXES

    OUTPUT:
        f: figure handle
    """
    h_tau = sol['h_tau']

    # allow for multiple types of car on one graph
    # (because outside option is in the same distribution!)
    f, ax = plt.subplots()
    plt.ion()

    dty = []
    for j in range(mp['ncartypes']):
        tmp = []
        for tau in range(mp['ntypes']):
            tmp.append(np.append(h_tau[tau][s['ipt']['car'][j]], [np.nan]))
        dty.append(tmp)

    dty = np.concatenate(dty, axis=0)

    # no car state - from h_tau
    qnc = 0
    qnocar = []
    for t in range(mp['ntypes']):
        qnocar.append(h_tau[t][s['ipt']['nocar']])  # outside option
    qnc = sum(qnocar)

    # split the no car fraction into many columns of height maxy
    maxy = max(np.sum(dty[:-1, :], axis=1))  # max fraction over types of consumer and car
    maxy = np.ceil(maxy * 1000) / 1000  # round up at 3rd digit
    nck = int(np.floor(qnc / maxy))  # number of columns of height maxy
    nckfrac = maxy * nck / qnc  # fraction of no car prob mass in repeated columns
    dty_nocar = np.tile(np.array(qnocar) * nckfrac / nck, (nck, 1))
    dty_nocar = np.vstack([dty_nocar, np.array(qnocar) * (1 - nckfrac)])

    # add to y values
    dty = np.vstack([dty, dty_nocar])

    # x ticks and labels
    dtx = np.arange(1, len(h_tau[0]) + nck + mp['ncartypes'] + 1)  # all distributions + nocar columns + separators
    # location of tick marks
    tickstep = 4
    xticks = np.arange(1, len(h_tau[0]) + 1, tickstep)  # cars
    xticks = np.append(xticks, len(h_tau[0]) + mp['ncartypes'] + int(nck / 2))  # no car

    # tick labels
    ticklabels = []
    if mp['ncartypes'] > 1:
        for j in range(mp['ncartypes']):
            lbl_used = ['%d' % i for i in range(tickstep, s['abar_j'][j] + 1, tickstep)]
            ticklabels.extend([mp['lbl_cartypes'][j]] + lbl_used)
    else:  # just one car
        ticklabels = ['%d' % i for i in range(0, s['abar_j'][0] + 1, tickstep)]
        # hotfix: delete the last tick to avoid clash with 'No car'
        ticklabels[-1] = ''
    ticklabels.append('No car')

    ax.set_xlim([dtx[0] - 0.5, dtx[-1] + 0.5])
    ax.grid(True, which='both', axis='y')
    ax.tick_params(axis='x', length=0)
    ax.set_xticks(xticks)
    ax.set_xticklabels(ticklabels)
    ax.bar(dtx, dty, linewidth=1, width=1, align='center')

    l1 = ax.legend(mp['lbl_types'], loc='northeastoutside')
    ax.set_xlabel('Car age')
    # ax.set_title('Holdings distributions (start of period) σ={}, T={}, ρ={}'.format(mp['sigma'], mp['transcost'], mp['ptranscost']))
    ax.set_ylabel('Fraction of the population')
    plt.tight_layout()

    return f

def prices_with_hom(mp, s, sol):
    assert mp['ntypes'] == 2 and mp['ncartypes'] == 1, 'Only implemented for 2*1 economy'
    
    # prices in solution 
    pp = [mp['pnew'][0]] + sol['p']
    
    # compute equilibria for each of the 1-type economies 
    pp_onetype = compute_hom_equilibria(mp)
    
    f, ax = plt.subplots()
    aa = np.arange(mp['abar_j0'][0])
    
    ax.plot(aa, [pp] + pp_onetype, linewidth=2)  # Replace 2 with the appropriate linewidth value
    leg_ = ['Two-type equilibrium', f"One-type: {mp['lbl_types'][0]}", f"One-type: {mp['lbl_types'][1]}"]
    ax.legend(leg_)
    
    ax.set_ylabel('Price, 1000 DKK')
    ax.set_xlabel('Car age')
    ax.grid(True, which='both', axis='y')
    ax.set_box_aspect(False)
    
    # Apply any additional layout settings if needed
    # set_fig_layout_post(f)  # Placeholder for any layout adjustments
    
    plt.show()
    return f

def compute_hom_equilibria(mp):
    """
    For a 2-type economy, this solves the two 1-type equilibria and
    returns just the two abar price-vectors.
    
    OUTPUT:
        pp: (ncarages*ntypes) matrix of equilibrium prices (including the
        exogenous new car price) for a homogeneous equilibrium consisting
        only of consumers of type tau (in columns).
    """
    mp_ = mp.copy()  # backup for safe-keeping
    
    # hard-coded list of fields that have to be updated
    # NOTE: there could be more type-specific fields, just
    # not in what we are using in the paper currently
    vv = ['u_0', 'u_a', 'mum', 'u_even', 'u_a_sq', 'lbl_types']
    
    assert mp['modeltype'] == 'reducedform', 'Only implemented for modeltype == "reducedform"'
    assert mp['ntypes'] == 2, 'Only implemented for examples with two household types'
    assert mp['ncartypes'] == 1, 'Only implemented for one-car settings'
    
    # preallocate output 
    pp = np.full((mp['abar_j0'][0], mp['ntypes']), np.nan)
    
    for tau in range(2):
        # copy model parameters
        mp = mp_.copy()
        mp['ntypes'] = 1
        mp['tw'] = [1]
        
        # overwrite type-specific coefficients
        for v in vv:
            mp[v] = [mp_[v][tau]]
        
        # solve model
        mp = TrModel.update_mp(mp)
        sol_homo = Equilibrium.solve(mp)

        # store equilibrium prices from the solution
        pp[:, tau] = [mp['pnew'][0]] + sol_homo['p']
    
    return pp
