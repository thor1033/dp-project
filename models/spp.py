import numpy as np
from scipy.sparse import diags, csr_matrix
from .tr_model import TrModel
from .dpsolver import DPSolver

class SPP:
    verbosity = 0
    abar_max = 100
    solmeth = ['poly']  # solution method for spp DP problem

    @staticmethod
    def solve_spp(mp):
        """
        solve_spp: solves the social planning problem to determine the optimal scrapping threshold
        by either successive approximations or policy iteration, depending on the choice of solution method.
        """
        mp = TrModel.update_mp(mp)  # update dependent parameters

        abar = SPP.abar_max-1  # pick a large upper bound for the possible scrap age of a car
        abar_spp = [[0] * mp['ncartypes'] for _ in range(mp['ntypes'])]

        p_spp = [[None for _ in range(mp['ncartypes'])] for _ in range(mp['ntypes'])]
        w_spp = [[None for _ in range(mp['ncartypes'])] for _ in range(mp['ntypes'])]

        for car in range(mp['ncartypes']):
            for t in range(mp['ntypes']):
                v0 = np.zeros(abar)
                if SPP.solmeth[0] == 'poly':  # poly algorithm - start with successive approximations
                    w, _, dr, _ = DPSolver.poly(lambda V: SPP.bellman_spp(V,mp, t, car), v0,  None,mp['bet'])
                #elif SPP.solmeth[0] == 'policy':  # policy iteration solution
                #    w, dr = DPSolver.policy(lambda V: SPP.bellman_spp(mp, V, t, car), mp, v0)
                else:
                    raise ValueError('spp.solmeth must be either "poly" or "policy".')

                abar_spp[t][car] = np.min(np.where(dr == 1)) - 1  # -1 because social planner has new cars in state space
                if not np.any(dr[:-1] == 1):
                    abar_spp[t][car] = mp['abar_j0'][car]
                    print('Not optimal to replace at any age in grid choosing abar_spp[t][car]=mp.abar_j0[car] -> increase spp.abar_max)')

                w = w[:abar_spp[t][car] + 1]

                p_spp[t][car] = mp['pnew'][car] - (w[0] - w) / mp['mum'][t]
                w_spp[t][car] = w
                if SPP.verbosity > 0:
                    print(f'solved spp for cartype {car} consumer type={t} abar_spp={abar_spp[t][car]}')

        print(abar_spp)
        return abar_spp, p_spp, w_spp

    @staticmethod
    def bellman_spp(V,mp, t, car):
        """
        bellman_spp: Bellman equation for social planner
        """
        abar = len(V) - 1
        a = np.arange(abar)
        ia = a

        #print(V)
        # accident probabilities and utility
        apr = TrModel.acc_prob_j(mp, np.arange(abar + 1), car, abar)
        urep = TrModel.u_car(mp, 0, t, car) - mp['mum'][t] * (mp['pnew'][car] - mp['pscrap'][car])
        ukeep = TrModel.u_car(mp, a, t, car)
        vrep = urep + mp['bet'] * ((1 - apr[0]) * V[1] + apr[0] * V[-1])
        print(V)
        vkeep = ukeep + mp['bet'] * ((1 - apr[ia]) * V[1:] + apr[ia] * V[-1])

        V_new = V.copy()  # Create a copy to modify
        V_new[:-1] = np.maximum(vrep, vkeep)
        V_new[-1] = vrep

        # Policy function (indicator for replacement)
        P = np.hstack((vrep > vkeep, 1))  # always replace oldest car

        # Frechet derivative of bellman operator (needed for dpsolver.policy and dpsolver.poly)
        tpm = diags((1 - P[ia]) * (1 - apr[ia]), 1).toarray()  # next period car age one year older if not replaced and not in accident
        tpm[:, -1] += P * apr[0] + (1 - P) * apr  # next period car is clunker if replaced and in accident or not replace car and in accident
        tpm[:, 1] += P * (1 - apr[0])  # next period car is one year old if replaced and not in accident
        dV = mp['bet'] * tpm

        # Expect utility given policy rule (needed for dpsolver.policy)
        Eu = P * urep + (1 - P) * np.hstack((ukeep, urep))
        return V_new, P, dV, Eu

    @staticmethod
    def hom_prices(mp, abar, tp=None, car=None):
        """
        This function calculates equilibrium prices in the homogeneous model using the closed-form solution.
        Syntax:
            [P, a] = prices(mp, abar, tp, car) if abar is a scalar
            [P, a] = prices(mp, abar) if abar is a list
        """
        if isinstance(abar, list):
            abar_cell = abar
            CARS = range(mp['ncartypes'])
            CONSUMERS = range(mp['ntypes'])
        else:
            CARS = [car]
            CONSUMERS = [tp]
            abar_cell = [[0] * mp['ncartypes'] for _ in range(mp['ntypes'])]
            abar_cell[tp][car] = abar

        P = {}
        a = {}
        for car in CARS:
            pnew = mp['pnew'][car]
            pscrap = mp['pscrap'][car]

            for tau in CONSUMERS:
                abar = abar_cell[tau][car]
                mum = mp['mum'][tau]

                x = diags((TrModel.acc_prob_j(mp, np.arange(abar - 1), car, abar) - 1) * mp['bet'] - 1).toarray() \
                    + diags(np.ones(abar - 1), -1).toarray() \
                    + diags((1 - TrModel.acc_prob_j(mp, np.arange(1, abar - 1), car, abar)) * mp['bet'], 1).toarray()

                y = TrModel.u_car(mp, np.arange(abar - 1), tau, car) / mum \
                    - TrModel.u_car(mp, np.arange(1, abar), tau, car) / mum \
                    + mp['bet'] * pscrap * TrModel.acc_prob_j(mp, np.arange(abar - 1), car, abar) \
                    - mp['bet'] * pscrap * np.append(TrModel.acc_prob_j(mp, np.arange(1, abar - 1), car, abar), 1)
                y[0] -= pnew

                P[tau, car] = np.linalg.solve(x, y)
                a[tau, car] = np.arange(1, abar)

        if not isinstance(abar, list):
            return P[tp, car], a[tp, car]
        return P, a

    @staticmethod
    def price_graph(mp, tp=1, car=1, fignr=None):
        """
        price_graph: solve model at parameters mp and plot homogeneous consumer equilibrium prices
        Syntax:
            price_graph(mp, tp, car, fignr):
                make price_graph for type (tp, car) in figure with number fignr
            price_graph(mp, tp, car):
                make price_graph for type (tp, car) in new figure
            price_graph(mp):
                make price_graph for type (1, 1) in new figure
        """
        import matplotlib.pyplot as plt

        abar_spp, _, _ = SPP.solve_spp(mp)

        if fignr is not None:
            plt.figure(fignr)
        else:
            plt.figure()

        plt.rc('text', usetex=True)
        for abar in range(abar_spp[tp][car], abar_spp[tp][car] - 6, -1):
            p, a = SPP.hom_prices(mp, abar, tp, car)
            if abar == abar_spp[tp][car]:
                plt.plot(np.arange(abar + 1), [mp['pnew'][car]] + p.tolist() + [mp['pscrap'][car]], linewidth=3, color='r')
                plt.plot([0, abar_spp[tp][car] + 5], [mp['pscrap'][car], mp['pscrap'][car]], linewidth=3, color='k')
            else:
                plt.plot(np.arange(abar + 1), [mp['pnew'][car]] + p.tolist() + [mp['pscrap'][car]], linewidth=1)
            plt.xlabel('Age of car')
            plt.ylabel('Equilibrium price of car')
            plt.xlim([0, abar_spp[tp][car] + 5])
            plt.ylim([-40, mp['pnew'][car] * 1.1])
        plt.title(f'Equilibrium (and non-equilibrium) solutions (with $\\bar{{a}} \\le$ optimal scrap age)')
        plt.legend([f'Equilibrium price, abar={abar_spp[tp][car]}, type={tp}, car={car}', f'Scrap price (car={car})'])

        if fignr is not None:
            plt.figure(fignr + 1)
        else:
            plt.figure()

        plt.rc('text', usetex=True)
        for abar in range(abar_spp[tp][car], abar_spp[tp][car] + 6):
            p, a = SPP.hom_prices(mp, abar, tp, car)
            if abar == abar_spp[tp][car]:
                plt.plot(np.arange(abar + 1), [mp['pnew'][car]] + p.tolist() + [mp['pscrap'][car]], linewidth=3, color='r')
                plt.plot([0, abar_spp[tp][car] + 5], [mp['pscrap'][car], mp['pscrap'][car]], linewidth=3, color='k')
            else:
                plt.plot(np.arange(abar + 1), [mp['pnew'][car]] + p.tolist() + [mp['pscrap'][car]], linewidth=1)
            plt.xlabel('Age of car')
            plt.ylabel('Equilibrium price of car')
            plt.xlim([0, abar_spp[tp][car] + 5])
            plt.ylim([-40, mp['pnew'][car] * 1.1])
        plt.title(f'Equilibrium (and non-equilibrium) solutions (with $\\bar{{a}} >$ optimal scrap age)')
        plt.legend([f'Equilibrium price, abar={abar_spp[tp][car]}, type={tp}, car={car}', f'Scrap price (car={car})'])

# Assuming the TrModel and DPSolver classes/functions are defined elsewhere.
