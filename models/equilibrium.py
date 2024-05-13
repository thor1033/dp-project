import numpy as np
from scipy.optimize import fsolve
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from .tr_model import TrModel
from .spp import SPP
from .dpsolver import DPSolver

class Equilibrium:
    verbosity = 0
    tolerance = 1e-6

    @staticmethod
    def solve(mp, s=None, p0=None):
        mp = TrModel.update_mp(mp)  # update parameters dependencies

        if s is None:
            s = TrModel.index(mp)
        if p0 is None:
            p0 = Equilibrium.price_init(mp, s)

        for attempt in range(1, 3):
            sol = Equilibrium.solve_once(mp, s, p0)

            if mp.fixprices:
                return sol

            if np.max(np.abs(sol['ed'])) > Equilibrium.tolerance or sol['exitflag'] < 0:
                p0 = Equilibrium.price_init(mp, s)  # reset prices and use spp solution

                if Equilibrium.verbosity > 0:
                    print('Problem in the equilibrium solution: maximum absolute excess excess['is']', np.max(np.abs(sol['ed'])))
                    print('exitflag from fsolve:', sol['exitflag'])
                    if attempt < 2:
                        print('Re-trying using the social planning solution as initial guess for prices')
                    else:
                        print('No equilibrium solution found in attempt', attempt, ': neither initial price guess or social planning starting guess worked.')
            else:
                if Equilibrium.verbosity > 0:
                    print('Equilibrium solution found in attempt', attempt, ': maximum absolute excess excess['is']', np.max(np.abs(sol['ed'])))
                break

    @staticmethod
    def price_init(mp, s):
        """
        Initialize price vector, p0, using social planner solution of used car prices
        SYNTAX: [p0] = equilibrium.price_init(mp, s)
        """
        p0 = np.full(s['np'], np.nan)
        abar_spp, p_spp, w_spp = SPP.solve_spp(mp)

        # Use social planner solution as starting values for p0
        for j in range(mp['ncartypes']):
            abar_j_spp, tau_j = max((abar, idx) for idx, abar in enumerate(abar_spp[:, j]))
            if abar_j_spp > mp['abar_j0'][j]:
                print(f'WARNING: Social planner solution not consistent with mp.abar_j0: abar_j_spp={abar_j_spp} > mp.abar_j0={mp["abar_j0"][j]} \nTruncating price vector - consider increasing mp.abar_j0')

            pj_spp = np.ones(1000) * mp['pscrap'][j]
            pj_spp[:abar_j_spp-1] = p_spp[tau_j, j][1:abar_j_spp]
            p0[s['ip'][j]] = pj_spp[:mp['abar_j0'][j]-1]
        
        return p0

    @staticmethod
    def solve_once(mp, s, p0=None):
        mp = TrModel.update_mp(mp)  # update parameters dependencies
        
        if p0 is None:
            p0 = Equilibrium.price_init(mp, s)

        if Equilibrium.verbosity > 0:
            print(f'Solving equilibrium with s.ns={s.ns} car states, {mp.ncartypes} car types, and {mp.ntypes} consumer types')

        # Initialize ev0 (if empty or if dimensions does not match)
        # Need to be global.
        global ev0
        if not ev0 or len(ev0) != mp.ntypes or len(ev0[0]) != s.ns:
            ev0 = [[0] * s.ns for _ in range(mp.ntypes)]

        # Find the heterogeneous agent equilibrium for given abar_cell values and p0 starting values via Newton algorithm
        if mp.fixprices:
            p = p0
            exitflag = 1
            output = None
        else:
            p, _, exitflag, output = fsolve(lambda p: Equilibrium.edf(mp, s, p), p0, xtol=1e-10,maxfev=2000)

        ed, ded, sol = Equilibrium.edf(mp, s, p)
        sol['p'] = p
        sol['exitflag'] = exitflag
        sol['output'] = output

        return sol

    @staticmethod
    def edf(mp, s, p):
        """
        Evaluate excess demand function (ed), its derivatives (ded), and the solution structure (sol) at a given price vector, p.
        """
        global ev0

        # Step 1: excess demand and its subcomponents

        # Compute the extended physical transition matrices F
        F = TrModel.age_transition(mp, s)

        # Initialize space
        ed_tau = np.full((len(p), mp['ntypes']), np.nan)
        ev_tau = [None] * mp['ntypes']
        ccp_tau = [None] * mp['ntypes']
        ctp_tau = [None] * mp['ntypes']
        q_tau = [None] * mp['ntypes']
        delta_tau = [None] * mp['ntypes']
        deltaK_tau = [None] * mp['ntypes']
        deltaT_tau = [None] * mp['ntypes']
        delta_scrap = [None] * mp['ntypes']
        util_tau = [None] * mp['ntypes']
        ev_scrap_tau = [None] * mp['ntypes']
        ccp_scrap_tau = [None] * mp['ntypes']

        for t in range(mp['ntypes']):
            if mp['tw'][t] > 0:
                # Precompute utility
                util_tau[t], ev_scrap_tau[t], ccp_scrap_tau[t] = TrModel.utility(mp, s, t, p)

                # Step 1: solve the DP problem for the price vector p for each consumer type
                bellman = lambda ev: TrModel.bellman(mp, s, util_tau[t], F, ev)
                ev_tau[t], ccp_tau[t] = DPSolver.poly(bellman, ev0[t], None, mp['bet'])

                # Trade, keep and scrap transition matrices
                delta_tau[t], deltaK_tau[t], deltaT_tau[t], delta_scrap[t] = TrModel.trade_transition(mp, s, ccp_tau[t], ccp_scrap_tau[t])

                # State transition matrix
                ctp_tau[t] = delta_tau[t] @ F['notrade']

                # Calculate the type-specific car distributions
                #q_tau[t] = DPSolver.ergodic(ctp_tau[t])

                # Step 2: compute contributions to excess demand for each consumer type
                idx = np.concatenate(s['is']['car_ex_clunker'])
                ed_tau[:, t] = deltaT_tau[t][:, idx].T @ q_tau[t] - (1 - ccp_tau[t][idx, s['id']['keep']]) * (1 - ccp_scrap_tau[t][idx]).flatten() * q_tau[t][idx]

        # Cumulate excess demand for all relevant ages of used cars that can be purchased, over all types of consumers
        ed = ed_tau @ np.array(mp['tw'])

        # Update starting values for ev. (used for next evaluation of edf)
        ev0 = ev_tau

        #if nargout >= 2:
        #    # Equilibrium holdings distribution aggregated over consumers
        #    q = np.hstack(q_tau) @ np.array(mp['tw']).reshape(-1, 1)

        #    # Compute market shares and equilibrium holdings distribution post trade
        #    h_tau = Equilibrium.h_tau(mp, s, q_tau, delta_tau)
        #    marketshares = Equilibrium.marketshares(mp, s, h_tau)

        #    fields = ['p', 'ed', 'F', 'q_tau', 'q', 'ev_tau', 'ccp_tau', 'ccp_scrap_tau', 'ctp_tau', 'delta_tau', 'h_tau', 'marketshares']
        #    sol_values = [p, ed, F, q_tau, q, ev_tau, ccp_tau, ccp_scrap_tau, ctp_tau, delta_tau, h_tau, marketshares]
        #    sol = dict(zip(fields, sol_values))

        #    # Return excess demand and derivative of excess demand for the price vector p for each consumer type
        #    grad_dp = G_TrModel.partial_deriv(mp, s, sol, 'prices')
        #    ded = grad_dp['ed']

        #    return ed, ded, sol
        #else:
        return ed

    @staticmethod
    def solve_dp(mp, s, p):
        # solve_dp: solves for each consumer type's optimal trading strategy via DP at a given price vector
        # It first solves the consumer DP problem and then evaluates the choice probabilities

        # note ev0 is a global cell array that stores the last values of the expected value
        # function to provide a starting point for the solution of the consumer problem
        global ev0

        # Compute the physical transition probability matrices for all car types (conditional on trading / or not trading)
        F = TrModel.age_transition(mp, s)

        # Initialize space
        ev_tau = [None] * mp.ntypes
        ccp_tau = [None] * mp.ntypes

        for t in range(mp.ntypes):
            if mp.tw[t] > 0:
                # Precompute utility
                util_tau, ev_scrap_tau, ccp_scrap_tau = TrModel.utility(mp, s, t, p)

                # Solve the DP problem for the price vector p for each consumer type
                bellman = lambda ev: TrModel.bellman(mp, s, util_tau, F, ev)
                ev_tau[t], ccp_tau[t] = DPSolver.poly(bellman, ev0[t], None, mp.bet)

        # Update starting values for ev (used for next evaluation of edf)
        ev0 = ev_tau

        #return F, ev_tau, ccp_tau, ccp_scrap_tau
        return F, ev_tau, ccp_tau 

    @staticmethod
    def h_tau(mp, s, q_tau, delta_tau):
        # equilibrium.h_tau: distribution of ownership after trading
        h_tau = []
        for tau in range(mp.ntypes):
            qpt = delta_tau[tau].T @ q_tau[tau]
            h_tau_tau = np.zeros((s.ns, 1))
            for j in range(mp.ncartypes):
                h_tau_tau[s.ipt.new[j], 0] = mp.tw[tau] * qpt[s['is'].clunker[j]]
                h_tau_tau[s.ipt.used[j], 0] = mp.tw[tau] * qpt[s['is'].car_ex_clunker[j]]
                h_tau_tau[s.ipt.nocar, 0] = mp.tw[tau] * qpt[s['is'].nocar]
            h_tau.append(h_tau_tau)
        return h_tau

    @staticmethod
    def marketshares(mp, s, h_tau):
        # equilibrium.marketshares: market shares after trading
        marketshares = np.zeros((mp.ntypes, mp.ncartypes + 1))
        for tau in range(mp.ntypes):
            for j in range(mp.ncartypes):
                marketshares[tau, j] = np.nansum(h_tau[tau][s.ipt.car[j]])
            marketshares[tau, mp.ncartypes] = h_tau[tau][s.ipt.nocar]
        return marketshares

    @staticmethod
    def print(mp, s, p):
        # Print market shares of each car type and excess demand
        ed, _, sol = Equilibrium.edf(mp, s, p)

        for i in range(mp.ncartypes):
            print('market share of car type', i, np.sum(sol['q'][s['is'].car[i]]))

        print('market share of outside good:', sol['q'][s.ns])
        print('{:>15} {:>15}'.format('prices', 'excess demands'))
        for price, excess_demand in zip(p, sol['ed']):
            print('{:15g} {:15g}'.format(price, excess_demand))

        if np.max(np.abs(sol['ed'])) > Equilibrium.tolerance:
            print('Maximum value of excess demand is', np.max(sol['ed']),
                  'possibility that this['is'] not an equilibrium')

    tolerance = 1e-6  # solution tolerance: equilibrium not considered solved if max(abs(ed)) > equilibrium.tolerance
