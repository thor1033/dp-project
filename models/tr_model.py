import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from collections import defaultdict

class TrModel:

    @staticmethod
    def setparams(mp0=None):
        """
        Standard parameters of the model
        SYNTAX :
          mp = trmodel.setparams()       # set all parameters to zeros
          mp = trmodel.setparams(mp0)    # update parameters with parameters in mp0

        NOTE: When adding new parameters to trmodel, they must be added here 
        DO NOT ENTER VALUES OF PARAMETERS - JUST ZEROS (or values that turn off the parameter)
        See also setparams.default to get default parameters
        """
        mp = {}

        # ******************************************
        # switches
        # ******************************************
        mp['es'] = 1  # use model with endogenous scrapage if 1 
        mp['fixprices'] = 0  # set to zero, to compute model solution at a given set of prices without solving for equilibrium 
        mp['ll_scrap'] = True  # add scrap decisions to likelihood if true

        # ******************************************
        # misc. parameters
        # ******************************************
        mp['dk_population'] = 2.5  # number of Danish households in millions

        # ******************************************
        # consumer types and car types
        # ******************************************
        # Some parameters below are consumer type specific (in rows) and car type specific (in columns) 
        # If number of entries is smaller than ntypes, last entry is repeated)

        mp['lbl_cartypes'] = ['Luxury', 'Normal']  # car type labels (for plotting)
        mp['abar_j0'] = [25, 25]  # car specific value of abar 
        mp['ncartypes'] = len(mp['abar_j0'])  # number of car types 

        mp['lbl_types'] = ['Rich', 'Poor']  # consumer type labels (for plotting)
        mp['tw'] = [0.5, 0.5]  # distribution of the types in the population (tw vector must sum to 1)
        mp['ntypes'] = len(mp['tw'])  # number of types of consumers

        # ******************************************
        # discount factor (beta)
        # ******************************************
        mp['bet'] = 0.95

        # ******************************************
        # accident parameters (alpha)
        # ap=1./(1+exp(-apv)), where apv = mp.acc_0{ct} + mp.acc_a{ct}.*a + mp.acc_even{ct}*(1-mod(a,2));     
        # ******************************************
        mp['acc_0'] = [-10]
        mp['acc_a'] = [0]
        mp['acc_even'] = [0]

        # ******************************************
        # parameters of GEV distribution (theta)
        # ******************************************
        mp['sigma'] = 1  # scale value of extreme value taste shocks in consumer problem at top level of the choice tree
        mp['sigma_s'] = 1  # the extreme value scale parameter for the idiosyncratic extreme value shocks affecting the sale vs scrappage decision

        # ******************************************
        # utility parameters (theta)
        # ******************************************
        mp['mum'] = [1]  # marginal utility of money

        mp['phi'] = [0.01]  # scale of driving utility (set low to exclude driving)

        # transactions costs parameters 
        mp['transcost'] = 0  # fixed transaction cost for buying cars
        mp['ptranscost'] = 0  # proportional component of transaction costs, as a fraction of the price of the car the consumer buys
        mp['psych_transcost'] = [0]  # utility cost of buying a car
        mp['psych_transcost_nocar'] = [0]  # additional utility cost of buying a car, when not owning a car
        mp['nocarsc'] = 0  # additional search cost incurred by a consumer who has no car
        mp['tc_sale'] = 0  # set the fixed component of the transaction cost to a seller (inspection/repair cost to make a car qualified to be resold)
        mp['tc_sale_age'] = 0  # coefficient of age on the total sales costs
        mp['ptc_sale'] = 0  # set the proportional component of the transaction cost to a seller (inspection/repair cost to make a car qualified to be resold)
        mp['tc_sale_even'] = 0  # incremental sales transactions costs during inspection years (even years) necessary to pass inspection before car can be sold

        # car utility parameters (see also trmodel.u_car) 
        # Reduced form car utility mp.u_0{tp, car}+mp.u_a{tp, car}*car_age + mp.u_a_sq{tp, car}*car_age.^2
        mp['u_0'] = [0]  # intercept in utility function (ntypes x ncartypes cell) 
        mp['u_a'] = [0]  # slope coefficient on car-age in utility function (ntypes x ncartypes cell) 
        mp['u_a_sq'] = [0]  # car age squared 

        mp['convexutility'] = 0  # this is a switch, if 1 then mp.u_a_sq is forced to be positive via the transformation exp(mp.u_a_sq) in u_car

        mp['u_og'] = [0]  # utility of outside good  (ntypes x 1 cell) 
        mp['u_even'] = [0]  # utility during even inspection years (expect to be a disutility or negative value due to the disutility of having car inspected)        

        # tax policy parameters (must be before specification of prices of fuel and cars)
        mp['vat'] = 0  # value added tax
        mp['cartax_lo'] = 0  # registration tax (below kink, K_cartax_hi)
        mp['cartax_hi'] = 0  # registration tax (above kink, K_cartax_hi)
        mp['tax_fuel'] = 0  # proportional fuel tax 
        mp['K_cartax_hi'] = 0  # mp.K_cartax_hi before mp.cartax_hi tax kicks in

        # ******************************************
        # Prices before taxes (prices after taxes are computed when solving the model)
        # if car or consumer type parameters have size smaller than ncartypes and ntypes, the last element is repeated.
        # ******************************************
        mp['pnew_notax'] = [100]  # new car prices (cartype specific)
        mp['pscrap_notax'] = [0]  # scrap car prices (cartype specific)
        mp['p_fuel_notax'] = 5  # price of fuel (DKK/liter) from average fuel price before tax
        # mp['p_fuel'] = 10.504  # price of fuel (DKK/liter) from average fuel price in 2008 from our dataset

        # ******************************************
        # Parameters of reduced form driving equation 
        # x = mp.db.pkm{tp}*pkm{car} + mp.db.car{car} + mp.db.tau{tp} + mp.db.a1{tp}*a + mp.db.a2{tp}*a.^2;
        # ******************************************
        # Reduced form driving equation coefficients are stored in db structure 
        mp['fe'] = [20]  # car specific fuel efficiency (km/liter)
        mp['db'] = {
            'specification': 'linear',  # 'linear' or 'log-log' 
            'pkm': [0],  # coefficient on pkm; 
            'car': [0],  # car type specific coefficient on car; 
            'tau': [0],  # coefficient with car type fixed effect by consumer type            
            'a1': [0],  # coefficient on a*1; 
            'a2': [0]  # coefficient on a^2; 
        }

        # ******************************************
        # Reduced form or structural form
        # ******************************************
        # Structural parameters can be identified from reduced form parameters for car demand and driving
        # by using trmodel.update_structural_par to obtain "structural" utility parameters mp.sp. 
        # Because of implicit dependence on type specific marginal utility of money, all these coefficients have to be consumer type specific
        # coefficient with price per kilometer by consumer type

        mp['modeltype'] = 'reducedform'
        # if mp.modeltype == 'reducedform': reduced form parameters mp.u_0, mp.u_a, and mp.u_a_sq are used in trmodel.u_car. 
        # if mp.modeltype == 'structuralform': structural parameters mp.sp, mp.mum are used in trmodel.u_car
        # 
        # If the model allows for driving, the reduced form parameters mp.u_0, mp.u_a, and mp.u_a_sq 
        # are not policy invariant since they on fuel-prices (and marginal utility of money and other structural parameters)
        # So to run counterfactuals where fuel prices are changed you need to set mp.modeltype = 'structuralform';
        #
        # Estimation strategy: 
        #   First set mp.modeltype = 'reducedform' to estimate reduced form parameters during estimation. 
        #   Then set mp.modeltype = 'structuralform' for running counter-factuals that changes fuel prices. 

        # ********************************************************************************
        # update parameters
        # ********************************************************************************
        # update mp with mp0 (default values are overwritten by input values mp0)
        if mp0 is not None:
            if 'db' in mp0:
                mp['db'].update(mp0['db'])
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

        mp['db']['car'] = TrModel.repcell(mp['db']['car'], mp['ncartypes'])
        mp['db']['pkm'] = TrModel.repcell(mp['db']['pkm'], mp['ntypes'])
        mp['db']['tau'] = TrModel.repcell(mp['db']['tau'], mp['ntypes'])
        mp['db']['a1'] = TrModel.repcell(mp['db']['a1'], mp['ntypes'])
        mp['db']['a2'] = TrModel.repcell(mp['db']['a2'], mp['ntypes'])

        # Car-specific parameters
        param_j = ['lbl_cartypes', 'abar_j0', 'acc_0', 'acc_a', 'acc_even', 'fe', 'pnew_notax', 'pscrap_notax']
        for param in param_j:
            mp[param] = TrModel.repcell(mp[param],mp['ncartypes'])

        # Household type-specific parameters
        param_tau = ['lbl_types', 'mum', 'phi', 'u_og', 'psych_transcost', 'psych_transcost_nocar']
        for param in param_tau:
            mp[param] = TrModel.repcell(mp[param], mp['ntypes'])

        # Car-and-household specific parameters
        param_tau_j = ['u_0', 'u_a', 'u_a_sq', 'u_even']
        for param in param_tau_j:
            mp[param] = TrModel.repcell(mp[param], mp['ntypes'])

        # The fundamental prices are p_fuel_notax, pnew_notax, plus the tax rates
        mp['p_fuel'], mp['pnew'], mp['pscrap'] = TrModel.price_notax(mp)

        if mp['ntypes'] == 1:
            mp['tw'] = 1

        if abs(sum(mp['tw']) - 1.0) > 1e-12:
            print('Error in trmodel.setup: mp.tw vector does not sum to 1')

        return mp
    @staticmethod
    def repcell(A, n, m=None):
        """
        Repeat elements of a list (or list of lists) to match the specified size.

        Parameters:
        A (list or list of lists): Input list or list of lists.
        n (int): Number of rows for the output list.
        m (int, optional): Number of columns for the output list. Defaults to None.

        Returns:
        list or list of lists: Output list or list of lists with repeated elements.
        """
        if m is None:
            n1 = len(A)
            if n1 > n:
                return A[:n]
            
            B = [None] * n
            B[:n1] = A
            if n1 < n:
                B[n1:n] = [A[-1]] * (n - n1)
            return B
        
        else:
            A = np.array(A, dtype=object)
            n1, m1 = A.shape
            B = np.empty((n, m), dtype=object)
            B[:min(n1, n), :min(m1, m)] = A[:min(n1, n), :min(m1, m)]
            
            if m1 < m:
                B[:, m1:m] = np.tile(B[:, m1-1:m1], (1, m - m1))
            
            if n1 < n:
                B[n1:n, :] = np.tile(B[n1-1:n1, :], (n - n1, 1))
            
            return B.tolist()

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
    def index(mp, abar_j=None):
        """
        Set up the state space for the TRModel.

        Parameters:
        mp (dict): Dictionary with model parameters.
        abar_j (list, optional): List of max ages for each car type. If not provided, defaults to mp['abar_j0'].

        Returns:
        dict: A dictionary representing the state space with various indices and metadata.
        """
        s = {}
        if abar_j is None:
            abar_j = [mp['abar_j0'][0]]

        s['abar_j'] = abar_j
        # Initialize state and decision indices
        ei = 0
        eip = 0
        s['is'] = {'car': [], 'car_ex_clunker': [], 'clunker': [], 'age': []}
        s['id'] = {'trade': [], 'trade_used': [], 'trade_new': [], 'age': []}
        s['ip'] = []

        for j, max_age in enumerate(abar_j):
            si = ei + 1
            ei += max_age

            # car owner state
            car_indices = list(range(si, ei + 1))
            s['is']['car'].append(car_indices)
            s['is']['car_ex_clunker'].append(car_indices[:-1])
            s['is']['clunker'].append(car_indices[-1])

            # trade decision
            trade_indices = [i + 1 for i in car_indices]
            s['id']['trade'].append(trade_indices)
            s['id']['trade_used'].append(trade_indices[1:])
            s['id']['trade_new'].append(trade_indices[0])

            # corresponding car age values
            s['is']['age'].extend(list(range(1, max_age + 1)))

            # indexes for used car prices
            sip = eip + 1
            eip += max_age - 1
            s['ip'].append(list(range(sip, eip + 1)))

        s['ns'] = ei + 1
        s['nd'] = ei + 2
        s['np'] = eip
        s['is']['nocar'] = s['ns']
        s['id']['purge'] = s['ns'] + 1

        s['is']['age'].append(np.nan)  # no car state age
        s['id']['age'] = s['is']['age'] + [np.nan] * 2  # keep and purge decision ages

        # Transition indices
        s['tr'] = {'choice': [], 'state': [i for sublist in s['is']['car'] for i in sublist] + [s['is']['nocar']],
                   'next_acc': [np.nan] * s['ns'], 'next_keep': [np.nan] * s['ns'],
                   'next_trade': [np.nan] * s['ns']}

        for idx, car in enumerate(s['is']['car']):
            for c in car:
                s['tr']['next_keep'][c - 1] = car[0] if c == car[-1] else c + 1
                s['tr']['next_acc'][c - 1] = s['is']['clunker'][idx]

        s['tr']['choice'].append(s['id']['purge'])
        s['tr']['next_keep'][s['is']['nocar'] - 1] = s['is']['nocar']
        s['tr']['next_acc'][s['is']['nocar'] - 1] = s['is']['nocar']
        
        
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
            uv = (mp['sp']['alpha_0'][tau][car] +
                  mp['sp']['alpha_a'][tau][car] * car_age +
                  mp['sp']['alpha_a_sq'][tau][car] * car_age ** 2 -
                  1 / (2 * mp['sp']['phi'][tau]) * 
                  (np.maximum(0, mp['sp']['gamma_0'][tau][car] + mp['sp']['gamma_a'][tau] * car_age - pkm * mp['mum'][tau])) ** 2)
        elif mp['modeltype'] == 'reducedform':
            if mp['convexutility']:
                uv = (mp['u_0'][tau][car] +
                      mp['u_a'][tau][car] * car_age +
                      np.exp(mp['u_a_sq'][tau][car]) * car_age ** 2)
            else:
                uv = (mp['u_0'][car] +
                      mp['u_a'][car] * car_age +
                      mp['u_a_sq'][car] * car_age ** 2)
        else:
            raise ValueError(f'Unexpected reduced form type, "{mp["modeltype"]}".')

        # add the (dis)utility from car inspections in even years 
        # (after the first inspection at age 4)
        inspection = (1 - car_age % 2) * (car_age >= 4)  # dummy for inspection year
        uv += mp['u_even'][car] * inspection

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
        sp['gamma_0'] = (-1*d0) * sp['phi']
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
