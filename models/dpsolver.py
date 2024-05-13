import numpy as np
from scipy.sparse import eye as speye
from scipy.linalg import solve
import time

class DPSolver:
    @staticmethod
    def setup(apopt=None):
        """
        DPSolver.setup: setup of algorithm solution parameters, ap, used in solve
        Syntax: ap = DPSolver.setup(apopt)
        
        INPUT:
        apopt (optional): If apopt is specified, default parameters will be overwritten with elements in apopt.
        
        OUTPUT:
        ap: algorithm parameter structure
        
        See also:
        dpsolver.sa, dpsolver.nk
        """
        # default values of ap
        ap = {
            'sa_max': 20,            # Maximum number of contraction steps
            'sa_min': 2,             # Minimum number of contraction steps
            'sa_tol': 1.0e-3,        # Absolute tolerance before (in DPSolver.poly: tolerance before switching to N-K algorithm)
            'max_fxpiter': 35,       # Maximum number of times to switch between Newton-Kantorovich iterations and contraction iterations.
            'pi_max': 40,            # Maximum number of Newton-Kantorovich steps
            'pi_tol': 1.0e-13,       # Final exit tolerance in fixed point algorithm, measured in units of numerical precision
            'tol_ratio': 1.0e-03,    # Relative tolerance before switching to N-K algorithm
                                     # when discount factor is supplied as input in DPSolver.poly
            'printfxp': 0,           # Print iteration info for fixed point algorithm
                                     # ap.printfxp=0 (No printing), ap.printfxp=1 (Summary info), ap.printfxp>1 (Detailed info)
            'check_deriv': 0,        # Check analytical derivatives of bellman against numerical derivatives
            'check_deriv_tol': 1e-6  # Absolute tolerance before for difference between analytical and numerical derivatives of bellman operator
        }

        if apopt is not None:
            for key, value in apopt.items():
                ap[key] = value

        return ap

    @staticmethod
    def poly(bellman, V0, ap=None, bet=None):
        """
        DPSolver.poly: Solve for fixed point using a combination of Successive Approximations (SA) and Newton-Kantorovich (NK) iterations
        
        Syntax: [V, P, dV, iter] = DPSolver.poly(bellman, V0, ap, bet)
        
        INPUT:
            bellman: [V, P, dV] = Bellman equation with fixed point, V
                     Python function on the form [V, dV] = bellman(V)
                     where V is the value function (m x 1), P is the policy function 
                     and dV is the (m x m) Frechet derivative of the Bellman operator
            V0: Initial guess value function, V.
                [m x 1 matrix]
            ap: Algorithm parameters. See DPSolver.setup
            bet: Discount factor. Enters rule for stopping SA and switching to NK iterations.
                 SA should stop prematurely when relative tolerance is close to bet.
        
        OUTPUT:
            V: m x 1 matrix. Fixed point, V
            P: Policy function at fixed point
            dV: Frechet derivative of the Bellman operator
        """

        # Set default settings for fixed point algorithm, for ap's not given in input
        # (overwrites defaults if ap is given as input)
        if ap is None:
            ap = DPSolver.setup()
        else:
            ap = DPSolver.setup(ap)

        start_time = time.time()
        iter = []

        for k in range(ap['max_fxpiter']):  # poly-algorithm loop (switching between SA and N-K and back)

            # SECTION A: CONTRACTION ITERATIONS
            if ap['printfxp'] > 0:
                print('\nBegin contraction iterations (for the {}. time)'.format(k + 1))

            if bet is not None:
                V0, sa_iter = DPSolver.sa(bellman, V0, ap, bet)
            else:
                V0, sa_iter = DPSolver.sa(bellman, V0, ap)
            iter.append({'sa': sa_iter})

            # SECTION B: NEWTON-KANTOROVICH ITERATIONS
            if ap['printfxp'] > 0:
                print('\nBegin Newton-Kantorovich iterations (for the {}. time)'.format(k + 1))

            V0, P, dV, nk_iter = DPSolver.nk(bellman, V0, ap)
            iter[-1]['nk'] = nk_iter

            if nk_iter['converged']:
                if ap['printfxp'] > 0:
                    print('Convergence achieved!\n\n')
                    print('Elapsed time: {:.5f} seconds'.format(time.time() - start_time))
                break  # out of poly-algorithm loop
            else:
                if k >= ap['max_fxpiter'] - 1:
                    print('Warning: DPSolver.poly: Non-convergence! Maximum number of iterations exceeded without convergence!')
                    break  # out of poly-algorithm loop with no convergence

        V = V0
        return V, iter

    @staticmethod
    def sa(bellman, V0, ap=None, bet=None):
        """
        DPSolver.sa: Solve for fixed point using successive approximations
        
        Syntax: [V, iter] = DPSolver.sa(bellman, V0, ap, bet)
        
        INPUT:
            bellman: V = Bellman equation with fixed point, V
                     Python function on the form V = bellman(V)
                     where V is the value function (m x 1)
            V0: Initial guess value function, V.
                [m x 1 matrix]
            ap: Algorithm parameters. See DPSolver.setup
            bet: Discount factor. Enters rule for stopping SA and switching to NK iterations.
                 SA should stop prematurely when relative tolerance is close to bet.
        
        OUTPUT:
            V: m x 1 matrix. Approximation of fixed point, V
            iter: Dictionary containing iteration details
        """

        # Set default settings for fixed point algorithm, for ap's not given in input
        # (overwrites defaults if ap is given as input)
        ap = DPSolver.setup(ap)

        start_time = time.time()
        iter = {
            'tol': np.nan * np.ones(ap['sa_max']),
            'rtol': np.nan * np.ones(ap['sa_max']),
            'converged': False
        }

        for i in range(ap['sa_max']):
            V = bellman(V0)
            iter['tol'][i] = np.max(np.abs(V - V0))
            iter['rtol'][i] = iter['tol'][i] / iter['tol'][max(i-1, 0)]
            V0 = V  # accept SA step and prepare for new iteration

            # Stopping criteria
            if bet is not None:
                if (i >= ap['sa_min']) and (np.abs(bet - iter['rtol'][i]) < ap['tol_ratio']):
                    iter['message'] = 'SA stopped prematurely due to relative tolerance. Start NK iterations'
                    break

            # Rule 2:
            adj = np.ceil(np.log10(np.abs(np.max(V0))))
            ltol = ap['sa_tol'] * 10 ** adj  # Adjust final tolerance
            ltol = ap['sa_tol']
            if (i >= ap['sa_min']) and (iter['tol'][i] < ltol):
                iter['message'] = 'SA converged after {} iterations, tolerance: {:.10g}\n'.format(i + 1, iter['tol'][i])
                iter['converged'] = True
                break

        iter['n'] = i + 1
        iter['tol'] = iter['tol'][:i + 1]
        iter['rtol'] = iter['rtol'][:i + 1]
        iter['time'] = time.time() - start_time

        DPSolver.print_iter(iter, ap)

        return V, iter

    @staticmethod
    def policy(bellman, V0, ap=None):
        """
        DPSolver.policy: Solve for fixed point using Newton-Kantorovich iterations

        Syntax: [V, P, dV, iter] = DPSolver.policy(bellman, V0, ap)

        INPUT:
            bellman: [V, P, dV] = Bellman equation with fixed point, V
                     Python function on the form [V, dV] = bellman(V)
                     where V is the value function (m x 1), P is the policy function
                     and dV is the (m x m) Frechet derivative of the Bellman operator
            V0: Initial guess value function, V.
                [m x 1 matrix]
            ap: Algorithm parameters. See DPSolver.setup

        OUTPUT:
            V: m x 1 matrix. Fixed point, V
            P: Policy function at fixed point
            dV: Frechet derivative of the Bellman operator
            iter: Dictionary containing iteration details
        """

        # Set default settings for fixed point algorithm, for ap's not given in input
        # (overwrites defaults if ap is given as input)
        ap = DPSolver.setup(ap)

        solution_time = time.time()
        iter = {
            'tol': np.full(ap['pi_max'], np.nan),
            'rtol': np.full(ap['pi_max'], np.nan),
            'converged': False
        }

        m = len(V0)
        for i in range(ap['pi_max']):  # do at most pi_max N-K steps
            # Do policy iteration step
            V1, P, dV, Eu = bellman(V0)  # also return value and policy function
            F = speye(m) - dV  # using dV from last call to bellman
            V = np.linalg.solve(F.toarray(), Eu)  # policy-iteration

            # tolerance
            iter['tol'][i] = np.max(np.abs(V - V0))
            V0 = V

            # adjusting the N-K tolerance to the magnitude of ev
            adj = np.ceil(np.log10(np.abs(np.max(V0))))
            ltol = ap['pi_tol'] * 10**adj  # Adjust final tolerance
            ltol = ap['pi_tol']  # tolerance

            if iter['tol'][i] < ltol:
                # Convergence achieved
                iter['message'] = f'Policy converged after {i+1} iterations, tolerance: {iter["tol"][i]:.10g}\n'
                iter['converged'] = True
                break

        iter['time'] = time.time() - solution_time
        iter['n'] = i + 1
        iter['tol'] = iter['tol'][:i+1]
        iter['rtol'] = iter['rtol'][:i+1]

        DPSolver.print_iter(iter, ap)

        return V, P, dV, iter

    @staticmethod
    def nk(bellman, V0, ap=None):
        """
        DPSolver.nk: Solve for fixed point using Newton-Kantorovich iterations

        Syntax: [V, P, dV, iter] = DPSolver.nk(bellman, V0, ap)

        INPUT:
            bellman: [V, P, dV] = Bellman equation with fixed point, V
                     Python function on the form [V, P, dV] = bellman(V)
                     where V is the value function (m x 1), P is the policy function
                     and dV is the (m x m) Frechet derivative of the Bellman operator
            V0: Initial guess value function, V.
                [m x 1 matrix]
            ap: Algorithm parameters. See DPSolver.setup

        OUTPUT:
            V: m x 1 matrix. Fixed point, V
            P: Policy function at fixed point
            dV: Frechet derivative of the Bellman operator
            iter: Dictionary containing iteration details
        """

        # Set default settings for fixed point algorithm, for ap's not given in input
        # (overwrites defaults if ap is given as input)
        ap = DPSolver.setup(ap)

        solution_time = time.time()
        iter = {
            'tol': np.full(ap['pi_max'], np.nan),
            'rtol': np.full(ap['pi_max'], np.nan),
            'converged': False
        }

        m = len(V0)
        for i in range(ap['pi_max']):  # do at most pi_max N-K steps
            # Do N-K step
            V1, P, dV = bellman(V0)  # also return value and policy function
            if ap['check_deriv']:
                df_nm, df_an, df_err = DPSolver.check_deriv(bellman, V0, ap)
                if np.max(np.abs(df_err)) > ap['check_deriv_tol']:
                    print('Warning: Max abs difference between analytical and numerical derivatives of bellman operator is larger than tolerance')

            F = speye(m) - dV  # using dV from last call to bellman
            V = V0 - solve(F.toarray(), V0 - V1)  # NK-iteration

            # do additional SA iteration for stability and accurate measure of error bound
            V0 = bellman(V)

            # tolerance 
            iter['tol'][i] = np.max(np.abs(V - V0))

            # adjusting the N-K tolerance to the magnitude of ev
            adj = np.ceil(np.log10(np.abs(np.max(V0))))
            ltol = ap['pi_tol'] * 10**adj  # Adjust final tolerance
            ltol = ap['pi_tol']  # tolerance

            if iter['tol'][i] < ltol:
                # Convergence achieved
                iter['message'] = f'N-K converged after {i+1} iterations, tolerance: {iter["tol"][i]:.10g}\n'
                iter['converged'] = True
                break

        iter['time'] = time.time() - solution_time
        iter['n'] = i + 1
        iter['tol'] = iter['tol'][:i+1]
        iter['rtol'] = iter['rtol'][:i+1]

        DPSolver.print_iter(iter, ap)

        return V, P, dV, iter

    @staticmethod
    def print_iter(iter, ap):
        if ap['printfxp'] > 1:  # print detailed output
            print('iter           tol        tol(j)/tol(j-1)')
            for i in range(len(iter['tol'])):
                print(f'{i+1:3d}   {iter["tol"][i]:16.8e} {iter["rtol"][i]:16.8f}')

        if ap['printfxp'] > 0:  # print final output
            if iter['converged']:
                print(f'{iter["message"]}')
            else:
                print(f'Maximum number of iterations reached, tolerance: {iter["tol"][-1]:.10g}')
            print(f'Elapsed time: {iter["time"]:.5f} seconds')

    @staticmethod
    def check_deriv(fun, x0, ap):
        """
        check_deriv: Check analytical derivatives of Bellman operator against numerical derivatives.

        Syntax: [df_nm, df_an, df_err] = DPSolver.check_deriv(fun, x0, ap)

        INPUT:
        fun: Bellman equation function on the form [V, P, dV] = fun(V)
        x0: Initial guess value function, V.
        ap: Algorithm parameters.

        OUTPUT:
        df_nm: Numerical derivative.
        df_an: Analytical derivative.
        df_err: Difference between analytical and numerical derivatives.
        """
        f_nm, _, df_an = fun(x0)
        df_nm = DPSolver.gradp(fun, x0, ap)
        df_err = df_an - df_nm
        return df_nm, df_an, df_err

    @staticmethod
    def gradp(fun, x0, ap):
        """
        gradp: Compute numerical gradient of a function using finite differences.

        Syntax: df_nm = DPSolver.gradp(fun, x0, ap)

        INPUT:
        fun: Function to differentiate.
        x0: Point at which to compute the gradient.
        ap: Algorithm parameters.

        OUTPUT:
        df_nm: Numerical gradient.
        """
        epsilon = 1e-5
        x0 = np.asarray(x0)
        f0, _, _ = fun(x0)
        df_nm = np.zeros((len(f0), len(x0)))

        for i in range(len(x0)):
            x0_eps = np.copy(x0)
            x0_eps[i] += epsilon
            f_eps, _, _ = fun(x0_eps)
            df_nm[:, i] = (f_eps - f0) / epsilon

        return df_nm
