import findMaxRpaNetworks as maxRPA
import networkAnalysisClass as network
import pickle


# define the interface reactions
def get_interface_reactions():
    # we need to specify the actuation reactions in this function
    # In the example, we have only one actuation X_2 -- > X_2 + X_4 (mRNA).
    # We only allow doubly-bimolecular reactions (i.e. both reactants/products are bimolecular). Each bimolecular
    # combination is specified by a two-dimensional vector (a,b) of non-negative integers denoting the species' indices.
    # Under this notation, the l.h.s. of the actuation reaction is given by the vector nu = (2, 0) and the r.h.s. vector
    # is nu' = (2, 4). The overall reaction is given by the list [nu, nu']. The set of all actuation reactions is a list
    # of such lists.

    # In our example, the actuation reaction list only consists of one list. More actuation reactions can be added if
    # needed.

    return [[(2, 0), (2, 4)]]


# define the controlled network reactions
def get_controlled_network_reactions():
    # We have three reactions in the gene-expression network
    # 1. (Protein Translation) X_4 --> X_4 + X_1  nu =(4,0) , nu' = (4, 1)
    # 2. (mRNA degradation) X_4 --> 0  nu = (4, 0), nu' = (0, 0)
    # 3. (protein degradation) X_1 --> 0  nu = (1, 0), nu' = (0, 0)

    return [[(4, 0), (4, 1)], [(4, 0), (0, 0)], [(1, 0), (0, 0)]]


def main(argv):
    numPlantSpecies = 2  # mention the number of species in the controlled network (i.e. 2 for gene-expression)
    num_IM_species = 2  # mention the number of species in the controller network (i.e. 2 in our example)
    k = 1  # selects top k networks in all three categories "stochastic_antithetic, deterministic_antithetic" and
    # "homothetic"
    filename = "BestNetworks_100.pkl"  # this is the filename where results are stored in the current folder
    include_stochastic = True  # set True if stochastic analysis is also required for the top k networks.
    shouldNewScreeningAnalysisBeConducted = False  # set True if a new scan is needed. Otherwise saved results are used
    # for visualisation.

    # Species arranged as follows: X_1 is the output species, then come the IM species, then the rest of the species
    numSpecies = num_IM_species + numPlantSpecies
    possible_q_vals = maxRPA.generatePossible_q_vals(num_IM_species, num_IM_species + 1)
    top_k_networks = {}
    typeOfDesigns = {}

    if shouldNewScreeningAnalysisBeConducted:
        TotalNetworksAnalysed = 0

        print(f"Total Number of q-vector values to be considered: {len(possible_q_vals)}")
        for q in possible_q_vals:
            TotalNetworksAnalysedForThis_q = 0
            first_two_reactions_list, rhs_list = maxRPA.generatePossibleFirstTwoReactions(num_IM_species + 1, q)
            internal_controller_reactions_list, internal_plant_reactions_list, controller_plant_interface_reactions_list = \
                maxRPA.generateNetworkReactions(num_IM_species + 1, q)
            interface_reactions = get_interface_reactions()
            plant_reactions = get_controlled_network_reactions()
            i = -1

            controllerTopologies = maxRPA.powerset(internal_controller_reactions_list)
            TotalToBeAnalysesForThis_q = len(first_two_reactions_list) * len(controllerTopologies)

            for first_two_reactions in first_two_reactions_list:
                i += 1
                for internal_controller_reactions_tup in controllerTopologies:
                    TotalNetworksAnalysed += 1
                    TotalNetworksAnalysedForThis_q += 1
                    internal_controller_reactions = list(internal_controller_reactions_tup)
                    # initialise a network class instance
                    parameter_list = [1.0] * (len(first_two_reactions) + len(internal_controller_reactions) +
                                              len(interface_reactions) + len(plant_reactions))
                    param_perturb_index = len(parameter_list) - 1
                    initial_state = [0.1] * numSpecies
                    print(
                        f"\nAnalysing network: {TotalNetworksAnalysed}, This is network: {TotalNetworksAnalysedForThis_q}/"
                        f"{TotalToBeAnalysesForThis_q} for q = {q}")
                    mrpa_network = network.networkAnalysis(numSpecies, initial_state, first_two_reactions,
                                                           internal_controller_reactions,
                                                           interface_reactions,
                                                           plant_reactions, q, rhs_list[i], parameter_list,
                                                           param_perturb_index)

                    if mrpa_network.network_type not in typeOfDesigns.keys():
                        typeOfDesigns[mrpa_network.network_type] = 0
                    else:
                        typeOfDesigns[mrpa_network.network_type] += 1

                    if not mrpa_network.IsStoichiometryMatrixFullRowRank:
                        print("The stoichiometry of this network is rank-deficient!")
                        continue
                    mrpa_network.deterministicPerturbationAnalysis()
                    # mrpa_network.visualizeNetwork(False, title=mrpa_network.network_type)
                    if mrpa_network.IsUnstable:
                        print("The network might be unstable!")
                        continue
                    if mrpa_network.fixedPointsAtBoundary:
                        print("This network may have fixed points at the boundary, i.e. some components are nearly "
                              "zero!")
                        continue
                    print("\n")

                    if mrpa_network.network_type not in top_k_networks.keys():
                        top_k_networks[mrpa_network.network_type] = [mrpa_network]
                    elif len(top_k_networks[mrpa_network.network_type]) < k:
                        top_k_networks[mrpa_network.network_type].append(mrpa_network)
                    else:
                        top_k_networks[mrpa_network.network_type].sort(key=lambda x: x.max_rpa_score, reverse=True)
                        if mrpa_network.max_rpa_score >= top_k_networks[mrpa_network.network_type][k - 1].max_rpa_score:
                            del top_k_networks[mrpa_network.network_type][-1]
                            top_k_networks[mrpa_network.network_type].append(mrpa_network)
        # time for including stochastic analysis as well
        numStochasticTrajectories = 10000
        final_time = None
        for key in top_k_networks.keys():
            for k1 in range(len(top_k_networks[key])):
                if include_stochastic:
                    top_k_networks[key][k1].stochasticPerturbationAnalysis(numStochasticTrajectories, final_time)
        data_file = open(filename, "wb")
        pickle.dump([top_k_networks, typeOfDesigns, include_stochastic], data_file)
    else:
        data_file = open(filename, "rb")
        [top_k_networks, typeOfDesigns, include_stochastic] = pickle.load(data_file)
    totalNetworks = 0
    for key in typeOfDesigns.keys():
        totalNetworks += typeOfDesigns[key]
        print(f"Networks of type {key} analysed: {typeOfDesigns[key]}")
    print(f"Total Number of Networks: {totalNetworks}")
    for key in top_k_networks.keys():
        for k1 in range(min(len(top_k_networks[key]), k)):
            top_k_networks[key][k1].visualizeNetwork(include_stochastic, title=key)


if __name__ == '__main__':
    main(None)
