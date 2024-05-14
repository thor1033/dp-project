import matplotlib.pyplot as plt
from models.set_params import SetParams
from models.tr_model import TrModel
from models.equilibrium import Equilibrium
from utilities.graphs import Graphs

def main():
    # Clear and set up the environment
    plt.close('all')
    print("Running illustrations...")

    # Load default parameters
    params = SetParams.default()
    params['ntypes'] = 2
    params['ncartypes'] = 1
    params['lbl_cartypes'] = ['']

    # Initialize the model with parameters
    model = TrModel.setparams()
    s = model.index()

    # Low transaction cost scenario
    params['transcost'] = 0
    params = model.update_mp(params)

    # Solve the model in the baseline scenario
    equilibrium_solver = Equilibrium(params, s)
    sol = equilibrium_solver.solve()

    # Graphical output for low transaction cost
    grapher = Graphs(params)
    f = grapher.agg_holdings(sol)
    plt.ylim(plt.gca().get_ylim())
    plt.legend().set_visible(False)
    f.savefig('results/illustration/example_holdings_tc_0.eps', format='epsc')

    f1 = grapher.prices_with_hom(sol)
    plt.legend().set_visible(False)
    f1.savefig('results/illustration/example_prices_tc_0.eps', format='epsc')

    # High transaction cost scenario
    params['transcost'] = 10
    params['psych_transcost'] = [0]
    params = model.update_mp(params)

    sol = equilibrium_solver.solve()

    # Graphical output for high transaction cost
    f = grapher.agg_holdings(sol)
    plt.ylim(plt.gca().get_ylim())
    plt.legend().set_location('north')
    f.savefig('results/illustration/example_holdings_tc_10.eps', format='epsc')

    f2 = grapher.prices_with_hom(sol)
    f2.savefig('results/illustration/example_prices_tc_10.eps', format='epsc')

if __name__ == "__main__":
    main()
