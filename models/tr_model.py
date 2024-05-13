import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

class TrModel:

    @staticmethod
    def setparams(mp0=None):
        mp = {}

        # ******************************************
        # switches
        # ******************************************
        mp['es'] = 1
        mp['fixprices'] = 0
        mp['ll_scrap'] = True

        # ******************************************
        # misc. parameters
        # ******************************************
        mp['dk_population'] = 2.5

        # ******************************************
        # consumer types and car types
        # ******************************************
        mp['lbl_cartypes'] = ['Luxury', 'Normal']
        mp['abar_j0'] = [25, 25]
        mp['ncartypes'] = len(mp['abar_j0'])

        mp['lbl_types'] = ['Rich', 'Poor']
        mp['tw'] = [0.5, 0.5]
        mp['ntypes'] = len(mp['tw'])

        # ******************************************
        # discount factor (beta)
        # ******************************************
        mp['bet'] = 0.95

        # ******************************************
        # accident parameters (alpha)
        # ******************************************
        mp['acc_0'] = [-10]
        mp['acc_a'] = [0]
        mp['acc_even'] = [0]

        # ******************************************
        # parameters of GEV distribution (theta)
        # ******************************************
        mp['sigma'] = 1
        mp['sigma_s'] = 1

        # ******************************************
        # utility parameters (theta)
        # ******************************************
        mp['mum'] = [1]
        mp['phi'] = [0.01]

        mp['transcost'] = 0
        mp['ptranscost'] = 0
        mp['psych_transcost'] = [0]
        mp['psych_transcost_nocar'] = [0]
        mp['nocarsc'] = 0
        mp['tc_sale'] = 0
        mp['tc_sale_age'] = 0
        mp['ptc_sale'] = 0
        mp['tc_sale_even'] = 0

        mp['u_0'] = [0]
        mp['u_a'] = [0]
        mp['u_a_sq'] = [0]
        mp['convexutility'] = 0

        mp['u_og'] = [0]
        mp['u_even'] = [0]

        # tax policy parameters
        mp['vat'] = 0
        mp['cartax_lo'] = 0
        mp['cartax_hi'] = 0
        mp['tax_fuel'] = 0
        mp['K_cartax_hi'] = 0

        # ******************************************
        # Prices before taxes
        # ******************************************
        mp['pnew_notax'] = [100]
        mp['pscrap_notax'] = [0]
        mp['p_fuel_notax'] = 5

        # ******************************************
        # Parameters of reduced form driving equation
        # ******************************************
        mp['fe'] = [20]
        mp['db'] = {
            'specification': 'linear',
            'pkm': [0],
            'car': [0],
            'tau': [0],
            'a1': [0],
            'a2': [0]
        }

        # ******************************************
        # Reduced form or structural form
        # ******************************************
        mp['modeltype'] = 'reducedform'

        # update parameters
        if mp0 is not None:
            mp['db'].update(mp0.get('db', {}))
            for key in mp0:
                if key != 'db':
                    mp[key] = mp0[key]

        # update endogenous parameters
        mp = TrModel.update_mp(mp)
        mp['p_fuel_notax'], mp['pnew_notax'], mp['pscrap_notax'] = TrModel.price_notax(mp)
        
        return mp

    @staticmethod
    def update_mp(mp):
        # If mp.u_even is not a list, convert it to a list
        if not isinstance(mp['u_even'], list):
            mp['u_even'] = [mp['u_even']]

        # Update model dependent parameters when changing mp.ntypes or mp.ncartypes
        # if lists are smaller than mp.ntypes or mp.ncartypes the last element is repeated

        mp['db']['car'] = TrModel.repcell(mp['db']['car'], 1, mp['ncartypes'])
        mp['db']['pkm'] = TrModel.repcell(mp['db']['pkm'], mp['ntypes'], 1)
        mp['db']['tau'] = TrModel.repcell(mp['db']['tau'], mp['ntypes'], 1)
        mp['db']['a1'] = TrModel.repcell(mp['db']['a1'], mp['ntypes'], 1)
        mp['db']['a2'] = TrModel.repcell(mp['db']['a2'], mp['ntypes'], 1)

        # Car-specific parameters
        param_j = ['lbl_cartypes', 'abar_j0', 'acc_0', 'acc_a', 'acc_even', 'fe', 'pnew_notax', 'pscrap_notax']
        for param in param_j:
            mp[param] = TrModel.repcell(mp[param], 1, mp['ncartypes'])

        # Household type-specific parameters
        param_tau = ['lbl_types', 'mum', 'phi', 'u_og', 'psych_transcost', 'psych_transcost_nocar']
        for param in param_tau:
            mp[param] = TrModel.repcell(mp[param], mp['ntypes'], 1)

        # Car-and-household specific parameters
        param_tau_j = ['u_0', 'u_a', 'u_a_sq', 'u_even']
        for param in param_tau_j:
            mp[param] = TrModel.repcell(mp[param], mp['ntypes'], mp['ncartypes'])

        # The fundamental prices are p_fuel_notax, pnew_notax, plus the tax rates
        mp['p_fuel'], mp['pnew'], mp['pscrap'] = TrModel.price_aftertax(mp)

        if mp['ntypes'] == 1:
            mp['tw'] = 1

        if abs(sum(mp['tw']) - 1.0) > 1e-12:
            print('Error in trmodel.setup: mp.tw vector does not sum to 1')

        return mp
    @staticmethod
    def repcell(array, rows, cols):
        # Replicate elements of an array
        if not isinstance(array, list):
            array = [array]
        return (array * cols)[:cols]

    @staticmethod
    def price_notax(mp):
        # This function computes prices before taxes given after tax prices and tax rates in mp
        
        # fuel prices before taxes
        p_fuel_notax = mp['p_fuel'] / (1 + mp['tax_fuel'])

        pnew_notax = [None] * mp['ncartypes']
        pscrap_notax = [None] * mp['ncartypes']
        
        for car in range(mp['ncartypes']):
            pnew_notax[car] = TrModel.pcar_notax(
                mp['pnew'][car], mp['K_cartax_hi'], mp['cartax_lo'], mp['cartax_hi'], mp['vat']
            )

            # scrap price before tax: not currently implemented 
            pscrap_notax[car] = mp['pscrap'][car]

        return p_fuel_notax, pnew_notax, pscrap_notax

    @staticmethod
    def pcar_notax(pcar, kink_threshold, cartax_low, cartax_high, vat):
        """
        pcar_notax(): new car prices before registration taxes and VAT.
        """
        assert cartax_low <= cartax_high, 'Low-bracket rate is above high-bracket rate: this sounds like an error (unless you are doing a fancy counterfactual, then comment this assertion out)'
        assert vat < 1.0, 'Expecting VAT = 0.25 (or at least < 1.0). Delete this assertion if you are trying a crazy counterfactual and know what you are doing.'

        # how much is paid if the pre-tax car price (with VAT) puts it precisely at the cutoff 
        price_paid_at_cutoff = (1 + cartax_low) * kink_threshold

        # compute the final inversion separately depending on whether we are above or below the cutoff 
        if pcar <= price_paid_at_cutoff:
            pnew_notax = pcar / ((1 + cartax_low) * (1 + vat))
        else:
            numer = pcar + (cartax_high - cartax_low) * kink_threshold
            denom = (1 + vat) * (1 + cartax_high)
            pnew_notax = numer / denom
        
        return pnew_notax

    @staticmethod
    def price_aftertax(mp):
        # price_aftertax(): This function computes prices after taxes given before tax prices and tax rates in mp

        p_fuel = mp['p_fuel_notax'] * (1 + mp['tax_fuel']) # scalar

        pnew = [None] * mp['ncartypes']
        pscrap = [None] * mp['ncartypes']
        
        for icar in range(mp['ncartypes']):
            pnew_notax = TrModel.pcar_after_passthrough(mp['pnew_notax'][icar], mp, icar)

            pnew[icar] = TrModel.pcar_aftertax(
                pnew_notax, mp['K_cartax_hi'], mp['cartax_lo'], mp['cartax_hi'], mp['vat']
            )

            pscrap[icar] = mp['pscrap_notax'][icar] # currently, no special treatment 

        return p_fuel, pnew, pscrap

    @staticmethod
    def pcar_aftertax(pcar_notax, K_cartax_hi, cartax_lo, cartax_hi, vat):
        """
        pcar_aftertax(): new car prices including all taxes.
        """
        assert cartax_lo <= cartax_hi, 'hi/low bracket rates reversed: this could be an error!'
        assert vat <= 1.0, 'VAT should be in [0;1] (unless you are doing a crazy large counterfactual)'

        if pcar_notax * 1.25 <= K_cartax_hi:
            # no top-tax will be paid
            pcar_incl_taxes = (1 + cartax_lo) * 1.25 * pcar_notax
        else:
            # price is in the top-bracket
            pcar_incl_taxes = (1 + cartax_hi) * 1.25 * pcar_notax - (cartax_hi - cartax_lo) * K_cartax_hi

        return pcar_incl_taxes

    @staticmethod
    def set_up_passthrough(mp_baseline, rate):
        passthrough = {
            'pnew_notax_baseline': mp_baseline['pnew_notax'],
            'pnew_baseline': mp_baseline['pnew'],
            'rate': rate,
            'cartaxes_baseline': {
                'K_cartax_hi': mp_baseline['K_cartax_hi'],
                'cartax_lo': mp_baseline['cartax_lo'],
                'cartax_hi': mp_baseline['cartax_hi'],
                'vat': mp_baseline['vat']
            }
        }
        return passthrough

    @staticmethod
    def pcar_after_passthrough(pcar_raw, mp, icar, DOPRINT=False):
        """
        pcar_after_passthrough(): adds a mechanical firm-response to the raw pre-tax price.
        """
        if 'passthrough' in mp:
            assert 1 <= icar <= mp['ncartypes']
            assert 'pnew_notax_baseline' in mp['passthrough']
            assert 'rate' in mp['passthrough']

            # 1. compute price that *would* have prevailed absent any changes in firm behavior 
            pcar_at_full_passthrough = TrModel.pcar_aftertax(pcar_raw, mp['K_cartax_hi'], mp['cartax_lo'], mp['cartax_hi'], mp['vat'])
            mp0 = mp['passthrough']['cartaxes_baseline']  # tax rates in the baseline
            pcar_baseline_full_pasthrough = TrModel.pcar_aftertax(pcar_raw, mp0['K_cartax_hi'], mp0['cartax_lo'], mp0['cartax_hi'], mp0['vat'])
            delta_tax = pcar_at_full_passthrough - pcar_baseline_full_pasthrough

            # 2. change in manufacturer price
            # NOTE: "-1" because manufacturers move *opposite* of the policy makers intended direction
            delta_firm_price = (-1) * (1 - mp['passthrough']['rate']) * delta_tax

            # 3. final price before taxes get applied
            pcar = pcar_raw + delta_firm_price

            if DOPRINT:
                print(f'At full passthrough: p0 = {pcar_baseline_full_pasthrough:.2f} to p1 = {pcar_at_full_passthrough:.2f}: implied delta tax = {delta_tax:.2f}')
                print(f'Requested passthrough = {mp["passthrough"]["rate"] * 100.0:.2f}% => delta p raw = {delta_firm_price:.2f}')
                print(f'Final result: p0 = {pcar_raw:.2f} -> p with passthrough = {pcar:.2f}')
        
        else:
            # nothing to do
            pcar = pcar_raw
        
        return pcar       


    @staticmethod
    def index(mp, abar_j=None):
        """
        This function sets up the state space for the trmodel.

        SYNTAX: s = index(mp, abar_j)

        INPUT:
            mp: dict with model parameters (see trmodel.setparams)
            abar_j: list with max age (forced scrap age) for each car j=1,..,mp['ncartypes']

        OUTPUT:
            s: dict with the following elements:
               ... (description of fields in the output dict s)
        """

        if abar_j is None:
            abar_j = mp['abar_j0']
        
        s = {}
        s['abar_j'] = abar_j

        # Initialize state and decision indexes
        s['id'] = {'keep': 1}
        s['is'] = {'car': [], 'car_ex_clunker': [], 'clunker': []}
        s['ip'] = []
        ei = 0
        eip = 0
        for j in range(mp['ncartypes']):
            # Indexes states and trade decisions (including new car)
            si = ei + 1
            ei = ei + s['abar_j'][j]

            # Car owner state
            s['is']['car'].append(np.arange(si, ei + 1).tolist())
            s['is']['car_ex_clunker'].append(np.arange(si, ei).tolist())
            s['is']['clunker'].append(ei)

            # Trade decision
            s['id'].setdefault('trade', []).append(np.arange(si + 1, ei + 2).tolist())
            s['id'].setdefault('trade_used', []).append(np.arange(si + 2, ei + 2).tolist())
            s['id'].setdefault('trade_new', []).append([si + 1])

            # Corresponding car age values
            s['is'].setdefault('age', [])[si:ei + 1] = np.arange(1, s['abar_j'][j] + 1).tolist()
            s['id'].setdefault('age', [])[si + 1:ei + 2] = np.arange(0, s['abar_j'][j]).tolist()

            # Indexes for used car prices and excess demand
            sip = eip + 1
            eip = eip + s['abar_j'][j] - 1
            s['ip'].append(np.arange(sip, eip + 1).tolist())

        s['ns'] = ei + 1  # number of states (add one for no car state)
        s['nd'] = ei + 2  # number of decisions (add one for keep and purge)
        s['np'] = eip  # number of prices

        s['is']['nocar'] = s['ns']  # nocar state
        s['id']['purge'] = s['ns'] + 1  # purge decision

        s['is'].setdefault('age', [])[s['is']['nocar']] = np.nan
        s['id'].setdefault('age', [])[s['id']['keep']] = np.nan
        s['id'].setdefault('age', [])[s['id']['purge']] = np.nan

        # Indices for trade transition matrix
        s['tr'] = {'choice': []}
        for j in range(mp['ncartypes']):
            s['tr']['choice'].extend(s['id']['trade'][j][1:] + s['id']['trade'][j][:1])
        s['tr']['choice'].append(s['id']['purge'])

        # Index for all rows and columns in delta
        s['tr']['state'] = sum(s['is']['car'], []) + [s['is']['nocar']]

        # Index for next period car state
        s['tr']['next_acc'] = [np.nan] * s['ns']
        s['tr']['next_keep'] = np.array(s['tr']['state']) + 1
        s['tr']['next_trade'] = np.array(s['tr']['state'])
        for j in range(mp['ncartypes']):
            s['tr']['next_acc'][s['is']['car'][j]] = s['is']['clunker'][j]
            s['tr']['next_keep'][s['is']['clunker'][j]] = s['is']['car'][j][0]
        s['tr']['next_keep'][s['is']['nocar']] = s['is']['nocar']
        s['tr']['next_acc'][s['is']['nocar']] = s['is']['nocar']

        # Age transition matrices conditional on accidents
        s['Q'] = {}
        s['Q']['no_acc'] = csr_matrix((np.ones(s['ns']), (s['tr']['state'], s['tr']['next_keep'])), shape=(s['ns'], s['ns']))
        s['Q']['acc'] = csr_matrix((np.ones(s['ns']), (s['tr']['state'], s['tr']['next_acc'])), shape=(s['ns'], s['ns']))
        s['dQ'] = s['Q']['acc'] - s['Q']['no_acc']
        s['F'] = {}
        s['F']['no_acc'] = csr_matrix((np.ones(s['ns']), (s['tr']['state'], s['tr']['state'])), shape=(s['ns'], s['ns']))
        s['F']['acc'] = s['Q']['acc']
        s['dF'] = s['F']['acc'] - s['F']['no_acc']

        # Indexes for post trade distribution (before aging)
        s['ipt'] = {'car': s['is']['car'], 'nocar': s['is']['nocar']}
        for j in range(mp['ncartypes']):
            s['ipt'].setdefault('new', []).append(s['ipt']['car'][j][0])
            s['ipt'].setdefault('used', []).append(s['ipt']['car'][j][1:])
        s['ipt']['age'] = np.array(s['is']['age']) - 1

        return s
    @staticmethod
    def utility(mp, s, tau, price_vec):
        """
        utility: Function to compute utility for each consumer in each state and for each decision.
        """
        # update the prices in the vector price_vec to the cell array price_j
        price_j = [price_vec[s['ip'][i]] for i in range(mp['ncartypes'])]

        # net sales price after transactions cost 
        psell = TrModel.psell(mp, s, price_j) 

        # net purchase price after transactions cost 
        pbuy = TrModel.pbuy(mp, s, price_j) 

        u = np.full((s['ns'], s['nd']), np.nan)  # utility: number of states by number of decisions

        # utility of scrap option (when selling)
        ccp_scrap, ev_scrap = TrModel.p_scrap(mp, s, price_j, tau)

        for j in range(mp['ncartypes']):
            # utility of keeping (when feasible)
            u[s['is']['car_ex_clunker'][j], s['id']['keep']] = TrModel.u_car(mp, s['is']['age'][s['is']['car_ex_clunker'][j]], tau, j)
      
            # utility of trading 
            u[:, s['id']['trade'][j]] = (TrModel.u_car(mp, s['id']['age'][s['id']['trade'][j]], tau, j) +
                                         ev_scrap - mp['mum'][tau] * (pbuy[:, s['id']['trade'][j]] - psell) -
                                         mp['psych_transcost'][tau])

        # utility of purging (i.e. selling/scrapping the current car and not replacing it, i.e. switching to the outside good)
        u[:, s['id']['purge']] = mp['u_og'][tau] + mp['mum'][tau] * psell + ev_scrap

        # additional psych transactions cost and monetary search cost incurred by a consumer who has no car
        u[s['is']['nocar'], [item for sublist in s['id']['trade'] for item in sublist]] -= (
            mp['psych_transcost_nocar'][tau] + mp['mum'][tau] * mp['nocarsc'])

        # not possible to keep clunker or no car
        for clunker in s['is']['clunker']:
            u[clunker, s['id']['keep']] = np.nan
        u[s['is']['nocar'], s['id']['keep']] = np.nan

        return u, ev_scrap, ccp_scrap

    @staticmethod
    def p_scrap(mp, s, price_j, tau):
        """
        p_scrap: Function to compute expected utility of the option to scrap rather than sell.
        """
        psell = TrModel.psell(mp, s, price_j)

        pscrap = np.full(s['ns'], np.nan)
        for j in range(mp['ncartypes']):
            # tc = (price_j[j] * mp['ptc_sale'] + mp['tc_sale'] + mp['tc_sale_age'] * s['is']['age'][s['is']['car'][j]] + mp['tc_sale_even'] * (1 - s['is']['age'][s['is']['car'][j]] % 2))
            pscrap[s['is']['car_ex_clunker'][j]] = mp['pscrap'][j]

        ev_scrap = (mp['es'] == 1) * TrModel.logsum(np.column_stack((mp['mum'][tau] * (pscrap - psell), np.zeros(s['ns']))), mp['sigma_s'])
        ccp_scrap = 1 - np.exp(-ev_scrap / mp['sigma_s'])

        for j in range(mp['ncartypes']):
            ccp_scrap[s['is']['clunker'][j]] = 1
            ev_scrap[s['is']['clunker'][j]] = 0

        return ccp_scrap, ev_scrap

    @staticmethod
    def logsum(values, sigma):
        """
        Helper function to compute the log-sum.
        """
        max_val = np.max(values, axis=1)
        return max_val + sigma * np.log(np.sum(np.exp((values - max_val[:, None]) / sigma), axis=1))

    @staticmethod
    def psell(mp, s, price_j):
        """
        psell: Computes the selling price net of seller-side transactions costs.
        """
        psell = np.full(s['ns'], np.nan)
        for j in range(mp['ncartypes']):
            car_age = s['is']['age'][s['is']['car_ex_clunker'][j]]
            inspection = (1 - car_age % 2) * (car_age >= 4)  # dummy for inspection year
            psell[s['is']['car_ex_clunker'][j]] = price_j[j] * (1 - mp['ptc_sale']) - mp['tc_sale'] - mp['tc_sale_age'] * car_age - mp['tc_sale_even'] * inspection
            psell[s['is']['clunker'][j]] = mp['pscrap'][j]
        psell[s['is']['nocar']] = 0
        return psell

    @staticmethod
    def pbuy(mp, s, price_j):
        """
        pbuy: Computes the purchase price after transactions cost.
        """
        pbuy = np.full(s['nd'], np.nan)
        for j in range(mp['ncartypes']):
            pbuy[s['id']['trade_new'][j]] = mp['transcost'] + mp['pnew'][j] * (1 + mp['ptranscost'])
            pbuy[s['id']['trade_used'][j]] = mp['transcost'] + price_j[j] * (1 + mp['ptranscost'])
        pbuy[s['id']['purge']] = 0
        return pbuy
    @staticmethod
    def u_car(mp, car_age, tau, car):
        """
        u_car: Function to compute the utility of having a car.
        """
        pkm = mp['p_fuel'] / mp['fe'][car] * 1000  # p_fuel is measured in 1000 DKK/l, mp.fe{j} is measured as km/l, but pkm was in DKK/km in regression

        if mp['modeltype'] == 'structuralform':
            uv = (mp['sp']['alpha_0'][tau, car] +
                  mp['sp']['alpha_a'][tau, car] * car_age +
                  mp['sp']['alpha_a_sq'][tau, car] * car_age ** 2 -
                  1 / (2 * mp['sp']['phi'][tau]) * 
                  (np.maximum(0, mp['sp']['gamma_0'][tau, car] + mp['sp']['gamma_a'][tau] * car_age - pkm * mp['mum'][tau])) ** 2)
        elif mp['modeltype'] == 'reducedform':
            if mp['convexutility']:
                uv = (mp['u_0'][tau][car] +
                      mp['u_a'][tau][car] * car_age +
                      np.exp(mp['u_a_sq'][tau][car]) * car_age ** 2)
            else:
                uv = (mp['u_0'][tau][car] +
                      mp['u_a'][tau][car] * car_age +
                      mp['u_a_sq'][tau][car] * car_age ** 2)
        else:
            raise ValueError(f'Unexpected reduced form type, "{mp["modeltype"]}".')

        # add the (dis)utility from car inspections in even years 
        # (after the first inspection at age 4)
        inspection = (1 - car_age % 2) * (car_age >= 4)  # dummy for inspection year
        uv += mp['u_even'][tau][car] * inspection

        return uv
    @staticmethod
    def acc_prob_j(mp, a, ct, abar):
        """
        acc_prob_j: Probability of an accident as a function of age and type of vehicle.
        """
        f_apv = lambda mp: mp['acc_0'][ct] + mp['acc_a'][ct] * a + mp['acc_even'][ct] * (1 - a % 2)
        ap = 1 / (1 + np.exp(-f_apv(mp)))
        ap[a >= abar - 1] = 1.0

        #if 'dap' in locals():
        #    pvec0, mp = TrModel.estim_mp2pvec(mp, mp['pnames'])
        #    dap = ap * (1 - ap) * TrModel.estim_matgrad_mp(f_apv, mp, pvec0, 'alpha')
        #    return ap, dap

        return ap

    @staticmethod
    def age_transition(mp, s):
        """
        age_transition: Age transition probability matrices.
        """
        accprob = np.zeros(s['ns'])
        for j in range(mp['ncartypes']):
            accprob[s['is']['car'][j]] = TrModel.acc_prob_j(mp, np.arange(s['abar_j'][j]), j, s['abar_j'][j])

        F_notrade = s['Q']['no_acc'] + s['dQ'] * accprob[s['tr']['next_keep']]
        F_trade = s['F']['no_acc'] + s['dF'] * accprob[s['tr']['state']]
        return {'notrade': F_notrade, 'trade': F_trade, 'accprob': accprob}

    @staticmethod
    def trade_transition(mp, s, ccp, ccp_scrap=None):
        # Keeping transition probabilities
        deltaK = coo_matrix((ccp[:, s['id']['keep']], (np.arange(s['ns']), np.arange(s['ns']))), shape=(s['ns'], s['ns']))

        # Trading transition probabilities
        deltaT = ccp[s['tr']['state'], s['tr']['choice']]

        # Trade transition probability matrix
        delta = deltaT + deltaK.toarray()

        delta_scrap = None
        if ccp_scrap is not None:
            delta_scrap = (1 - ccp[:, s['id']['keep']]) * ccp_scrap

        return delta, deltaK, deltaT, delta_scrap

    @staticmethod
    def bellman(mp, s, util, F, ev):
        """
        bellman: Implements the Bellman operator ev1=Gamma(ev) for the consumer's problem of trading cars.
        """
        v = np.full((s['ns'], s['nd']), np.nan)  # choice specific value functions (states by decision)

        v[np.hstack(s['is']['car']), s['id']['keep']] = util[np.hstack(s['is']['car']), s['id']['keep']] + mp['bet'] * F['notrade'][np.hstack(s['is']['car']), :] @ ev

        v[:, np.hstack(s['id']['trade'])] = util[:, np.hstack(s['id']['trade'])] + (mp['bet'] * F['trade'][np.hstack(s['is']['car']), :] @ ev).T

        v[:, s['id']['purge']] = util[:, s['id']['purge']] + mp['bet'] * ev[s['is']['nocar']]
        v[np.isnan(util)] = np.nan
        ev1 = TrModel.logsum(v, mp['sigma'])

        #if 'ccp' in locals():
        #    ccp = np.exp((v - ev1[:, None]) / mp['sigma'])
        #    ccp[np.isnan(ccp)] = 0  # restore the nans in the elements of ccp_cell corresponding to infeasible choices

        #if 'dev' in locals():
        #    delta = TrModel.trade_transition(mp, s, ccp)
        #    dev = mp['bet'] * delta @ F['notrade']
        #    return ev1, ccp, dev

        return ev1

    @staticmethod
    def update_structural_par(mp):
        """
        update_structural_par: Updates structural parameters based on the model parameters.
        """
        sp = {}

        # 1. phi coefficients (on squared driving)
        mum = np.array(mp['mum'])
        sp['phi'] = mum / np.array(mp['db']['pkm'])

        # 2. compute gamma coefficients
        d0 = np.array(mp['db']['car']) + np.array(mp['db']['tau'])
        d1 = np.array(mp['db']['a1'])
        sp['gamma_0'] = -d0 * sp['phi']
        sp['gamma_a'] = -d1 * sp['phi']

        # 3. compute alpha coefficients
        pkm = mp['p_fuel'] / np.array(mp['fe']) * 1000
        if mp['convexutility']:
            sp['alpha_a_sq'] = np.exp(np.array(mp['u_a_sq'])) + 1 / (2 * sp['phi']) * (sp['gamma_a'] ** 2)
        else:
            sp['alpha_a_sq'] = np.array(mp['u_a_sq']) + 1 / (2 * sp['phi']) * (sp['gamma_a'] ** 2)
        sp['alpha_a'] = np.array(mp['u_a']) + sp['gamma_a'] / sp['phi'] * (sp['gamma_0'] - mum * pkm)
        sp['alpha_0'] = np.array(mp['u_0']) + 1 / (2 * sp['phi']) * (sp['gamma_0'] - mum * pkm) ** 2

        # 4. evaluate utility
        # uv = sp.alpha_0 + sp.alpha_a * car_age + sp.alpha_a_sq * car_age.^2 ...
        #     - 1/(2*sp.phi) * (sp.gamma_0 + sp.gamma_a * car_age - pkm * mum);

        # 5. check to make sure no negative driving: issue warnings if this is found
        for t in range(mp['ntypes']):
            for c in range(mp['ncartypes']):
                if sp['gamma_0'][t, c] + sp['gamma_a'][t] * (mp['abar_j0'][c] - 1) < mum[t] * pkm[c]:
                    print(f'Warning: update_structural_par negative driving predicted for household type {t} and cartype {c}')

        return sp
