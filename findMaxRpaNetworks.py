import numpy as np
import itertools


def create_complex_string(signature):
    signature_complex = ""
    for l in range(len(signature)):
        if signature[l] != 0:
            if signature_complex == "":
                signature_complex += f"X_{signature[l]} "
            elif l > 0 and signature[l] == signature[l - 1]:
                signature_complex = "2" + signature_complex
            else:
                signature_complex += f"+ X_{signature[l]} "
    if signature_complex == "":
        signature_complex = "0"
    return signature_complex


def create_complex_vector(signature, num_species):
    zeta = np.zeros(num_species)
    for l in range(len(signature)):
        if signature[l] != 0:
            zeta[signature[l] - 1] += 1
    return zeta


def check_compatibility(stoichiometric_vector, q):
    if np.inner(stoichiometric_vector, q) == 0:
        return True
    else:
        return False


def check_first_two_reactions(stoichiometric_first_two, q):
    if np.inner(stoichiometric_first_two[:, 0], q) > 0 and np.inner(stoichiometric_first_two[:, 1], q) < 0:
        return True
    else:
        return False


def generatePossible_q_vals(support_length_q, num_species):
    # support_length is the size of the IM
    d = num_species
    range_length_q = d
    values_q = np.concatenate((np.arange(-range_length_q, 0), np.arange(1, range_length_q + 1)))
    q_support_list = list(itertools.combinations_with_replacement(values_q, support_length_q))
    q_support_list = list({tuple(np.concatenate(([1], q[1:] / q[0]))) for q in q_support_list}) + \
                     list({tuple(np.concatenate(([-1], q[1:] / q[0]))) for q in q_support_list})
    return [np.concatenate(([0], q, np.zeros(d - support_length_q - 1))) for q in q_support_list]


def generatePossibleBimolecularReactions(d):
    species_labels = range(d + 1)
    possible_complexes = list(itertools.combinations_with_replacement(species_labels, 2))
    # return list(itertools.product(possible_complexes, possible_complexes))
    return list(itertools.permutations(possible_complexes, 2))


def generatePossibleFirstTwoReactions(numSpecies, q):
    species_labels = range(numSpecies + 1)
    possible_complexes = list(itertools.combinations_with_replacement(species_labels, 2))
    # generates all possible candidates for the first two reactions...note reactants for the two reactions would have
    # a common species and the sensing reaction would have an additional output species. It then check for
    # compatibility with q
    possible_first_two_reactions = list(itertools.product(species_labels, possible_complexes, possible_complexes))
    first_two_reactions_list = []
    linear_systems_rhs_list = []
    for reaction in possible_first_two_reactions:
        common_species_lhs = reaction[0]
        set_point_encoding_reaction_product = reaction[1]
        sensing_reaction_product = reaction[2]
        set_point_encoding_reaction_reactant = tuple([common_species_lhs])
        sensing_reaction_reactant = (1, common_species_lhs)
        zeta_set_point_encoding_reaction_reactant = create_complex_vector(set_point_encoding_reaction_product, numSpecies) \
                                                    - create_complex_vector(set_point_encoding_reaction_reactant, numSpecies)
        zeta_sensing_reaction = create_complex_vector(sensing_reaction_product, numSpecies) - \
                                create_complex_vector(sensing_reaction_reactant, numSpecies)
        zeta_first_two = np.stack((zeta_set_point_encoding_reaction_reactant, zeta_sensing_reaction), axis=-1)
        if check_first_two_reactions(zeta_first_two, q):
            # print(f"Reaction 1: {create_complex_string(set_point_encoding_reaction_reactant)} --->
            # {create_complex_string(set_point_encoding_reaction_product)}")
            # print(f"Reaction 2: {create_complex_string(sensing_reaction_reactant)} --->
            # {create_complex_string(sensing_reaction_product)}")
            first_two_reactions_list.append([[set_point_encoding_reaction_reactant,
                                              set_point_encoding_reaction_product],
                                             [sensing_reaction_reactant, sensing_reaction_product]])
            linear_systems_rhs_list.append([np.inner(zeta_set_point_encoding_reaction_reactant, q),
                                            np.inner(zeta_sensing_reaction, q)])
    return first_two_reactions_list, linear_systems_rhs_list


def generateNetworkReactions(d, q):
    internal_controller_reactions_list = []
    controller_plant_interface_reactions_list = []
    internal_plant_reactions_list = []
    im_species_indices = np.nonzero(q)
    plant_species_indices = np.where(q == 0)
    possible_reactions = generatePossibleBimolecularReactions(d)
    for reaction in possible_reactions:
        reaction_reactant = reaction[0]
        reaction_product = reaction[1]
        if reaction_reactant != reaction_product:
            nu2 = create_complex_vector(reaction_product, d)
            nu1 = create_complex_vector(reaction_reactant, d)
            zeta = nu2 - nu1
            # print(f"Stoichiometric vector is: {zeta}\n")
            if check_compatibility(zeta, q):
                if np.sum(nu1[plant_species_indices]) == 0 and np.sum(nu2[plant_species_indices]) == 0:
                    internal_controller_reactions_list.append([reaction_reactant, reaction_product])
                elif np.sum(nu1[im_species_indices]) == 0 and np.sum(nu2[im_species_indices]) == 0:
                    internal_plant_reactions_list.append([reaction_reactant, reaction_product])
                else:
                    controller_plant_interface_reactions_list.append([reaction_reactant, reaction_product])
    return internal_controller_reactions_list, internal_plant_reactions_list, controller_plant_interface_reactions_list


def powerset(controller_reactions):
    return list(itertools.chain.from_iterable(
        itertools.combinations(controller_reactions, k) for k in range(1, len(controller_reactions) + 1)))
