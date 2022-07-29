import numpy as np
import findMaxRpaNetworks as maxRPA
from scipy.integrate import odeint
import networkx as nx
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "text.latex.preamble": r'\usepackage{amsfonts}',
    "xtick.labelsize": 15,  # large tick labels
    'font.size': 15,
    'figure.figsize': [1600 / 72, 500 / 72]}  # default: 6.4 and 4.8
)
legend_font = font_manager.FontProperties(family='Arial',
                                          weight='bold',
                                          style='normal', size=16)


class networkAnalysis(object):
    def __init__(self, numSpecies, initial_state, first_two_reactions, internal_controller_reactions_list,
                 interface_reactions,
                 plant_reactions,
                 q, linear_system_rhs, param_list, param_perturb_index=None):
        self.final_time = 100  # species the final-time
        self.numTimeSamples = 1000  # this is the number of time-samples within each perturbation period
        self.numSpecies = numSpecies
        self.initial_state = initial_state
        self.initial_state_stochastic = np.ones(self.numSpecies) * 1
        self.first_two_reactions = first_two_reactions
        self.internal_controller_reactions_list = internal_controller_reactions_list
        self.interface_reactions = interface_reactions
        self.plant_reactions = plant_reactions
        self.q = np.zeros(numSpecies)
        self.q[:len(q)] = -q / linear_system_rhs[1]
        self.kappa = - linear_system_rhs[0] / linear_system_rhs[1]
        self.param_list = param_list
        self.param_perturb_index = param_perturb_index # parameter which is to be disturbed
        self.IsUnstable = False
        self.InstabilityThreshold = 1000  # set the threshold for instability
        # - the dynamics stops if one of the state-components reaches this threshold
        self.tolerance = 0.01  # for detecting zero (checking fixed-point components are at the boundary and if all
        # state-components are not changing at the steady-state)
        self.fixedPointsAtBoundary = False
        self.perturbation_times = [0.2 * self.final_time, 0.4 * self.final_time]  # the disturbance times
        self.perturbation_amounts = [-0.5, 0.5] # the relative disturbance amounts. Here x means parameter is changed
        # from theta to theta * (1 + x)
        self.allReactionsList = self.first_two_reactions + self.internal_controller_reactions_list \
                                + self.interface_reactions + self.plant_reactions
        self.numReactions = len(self.allReactionsList)
        self.S = np.zeros([self.numSpecies, self.numReactions], dtype=np.int8)
        self.ReactantMatrix = np.zeros([numSpecies, self.numReactions], dtype=np.int8)
        self.null_set = [(0, 0), (0,)]
        self.reference = None
        self.DeterministicOutputTrajectory = [None, None]
        self.StochasticMeanOutputTrajectory = [None, None, None]
        self.max_rpa_score = 0
        network_type = 1  # -2 for stochastic antithetic, -1 for deterministic antithetic, 1 for deterministic
        # homothetic
        if len(self.q[self.q < 0]) > 0 and len(self.q[self.q > 0]) > 0:
            network_type = -1

        nominal_perturbed_parameter_val = self.param_list[self.param_perturb_index]
        self.perturbed_param_vals = [nominal_perturbed_parameter_val]
        for p in self.perturbation_amounts:
            self.perturbed_param_vals.append(nominal_perturbed_parameter_val * (1 + p))

        i = 0
        print(f"\nConsidering the following network with {self.numSpecies} species")
        for [reaction_reactant, reaction_product] in self.allReactionsList:
            i += 1
            lhs = f"{maxRPA.create_complex_string(reaction_reactant)}"
            if reaction_reactant in self.null_set:
                lhs = f"0"
            rhs = f"{maxRPA.create_complex_string(reaction_product)}"
            if reaction_product in self.null_set:
                rhs = f"0"
            nu_1 = maxRPA.create_complex_vector(reaction_reactant, self.numSpecies)
            nu_2 = maxRPA.create_complex_vector(reaction_product, self.numSpecies)
            self.S[:, i - 1] = nu_2 - nu_1
            self.ReactantMatrix[:, i - 1] = nu_1
            print(f"Reaction {i}: {lhs} --> {rhs}")
        if max(self.ReactantMatrix[1:, 1]) == 0 and max(self.ReactantMatrix[:, 0]) == 0 and self.ReactantMatrix[
            0, 1] == 1:
            print("This network is stochastic maxRPA!")
            network_type = -2
        self.IsStoichiometryMatrixFullRowRank = (np.linalg.matrix_rank(self.S) == self.numSpecies)
        self.alpha = self.ReactantMatrix[0, 1] - self.ReactantMatrix[0, 0]

        if network_type == -2:
            self.network_type = "stochastic_antithetic"
        elif network_type == -1:
            self.network_type = "deterministic_antithetic"
        else:
            self.network_type = "homothetic"

    def propensity_function(self, state):
        x = state
        propensity = np.zeros(self.numReactions)
        for k in range(self.numReactions):
            propensity[k] = self.param_list[k]
            for i in range(self.numSpecies):
                if self.ReactantMatrix[i, k] > 0:
                    propensity[k] = propensity[k] * (x[i] ** self.ReactantMatrix[i, k])
        return propensity

    def stochastic_propensity_function(self, state):
        x = state
        propensity = np.zeros(self.numReactions)

        if self.IsUnstable or max(state) > self.InstabilityThreshold:
            self.IsUnstable = True
            return propensity

        for k in range(self.numReactions):
            propensity[k] = self.param_list[k]
            for j in range(self.numSpecies):
                for i in range(self.ReactantMatrix[j, k]):
                    propensity[k] *= float(state[j] - i)
                propensity[k] = propensity[k] / np.math.factorial(self.ReactantMatrix[j, k])
        return propensity

    def reaction_network_dydt(self, state, t):
        i = 0
        if self.IsUnstable or max(state) > self.InstabilityThreshold:
            self.IsUnstable = True
            return np.zeros([self.numSpecies])
        lambda_vec = self.propensity_function(state)
        return np.matmul(self.S, lambda_vec)

    def deterministicPerturbationAnalysis(self):
        # get the potential reference
        ref = (self.kappa * self.param_list[0] / self.param_list[0]) ** (1 / self.alpha)
        self.reference = ref
        L = len(self.perturbation_times)
        curr_state = self.initial_state
        curr_time = 0
        self.param_list[self.param_perturb_index] = self.perturbed_param_vals[0]
        for j in range(L + 1):
            if j == L:
                stop_time = self.final_time
            else:
                stop_time = self.perturbation_times[j]
            times = np.linspace(curr_time, stop_time, num=self.numTimeSamples)
            y = odeint(self.reaction_network_dydt, curr_state, times)
            if j < L:
                self.param_list[self.param_perturb_index] = self.perturbed_param_vals[j + 1]
            if j == 0:
                t = times
                out_states = y[:, 0]
            else:
                t = np.append(t, times)
                out_states = np.append(out_states, y[:, 0])
            curr_time = stop_time
            curr_state = y[-1, :]
            if len(curr_state[curr_state < self.tolerance]) > 0:
                self.fixedPointsAtBoundary = True
        # check if all the states are almost constant at the end (needed for stability)
        numberEndTimepoints = np.shape(y)[0]
        cutOffPoint = numberEndTimepoints * 8 // 10
        if np.mean(np.abs(np.mean(y[cutOffPoint:, :], axis=0) - curr_state)) > self.tolerance:
            self.IsUnstable = True
        # estimate the maxRPA score
        error_integral = 0
        for i in range(1, np.shape(t)[0]):
            error_integral += abs(out_states[i] - ref) * (t[i] - t[i - 1])
        error_integral = error_integral / self.final_time
        # error_integral = error_integral.item()
        max_rpa_Score = ref / (ref + error_integral) * 100
        self.DeterministicOutputTrajectory = [t, out_states]
        self.max_rpa_score = max_rpa_Score

    def gillespie_ssa_next_reaction(self, state):
        prop = self.stochastic_propensity_function(state)
        sum_prop = np.sum(prop)
        if sum_prop == 0:
            delta_t = np.math.inf
            next_reaction = -1
        else:
            prop = np.cumsum(np.divide(prop, sum_prop))
            delta_t = -np.math.log(np.random.uniform(0, 1)) / sum_prop
            next_reaction = sum(prop < np.random.uniform(0, 1))
        return delta_t, next_reaction

    def update_state(self, next_reaction, state):
        if next_reaction != -1:
            state = state + self.S[:, next_reaction]
        return state

    def run_gillespie_ssa(self, initial_state, stop_time):
        """
        Runs Gillespie's SSA without storing any values until stop_time; start time is 0 and
        initial_state is specified
        """
        t = 0
        state_curr = initial_state
        while 1:
            delta_t, next_reaction = self.gillespie_ssa_next_reaction(state_curr)
            t = t + delta_t
            if t > stop_time:
                return state_curr
            else:
                state_curr = self.update_state(next_reaction, state_curr)

    def generate_sampled_ssa_trajectory(self, final_time, num_time_samples):
        """
        Create a uniformly sampled SSA Trajectory.
        """
        L = len(self.perturbation_times)
        curr_state = self.initial_state_stochastic
        curr_time = 0
        self.param_list[self.param_perturb_index] = self.perturbed_param_vals[0]
        output_states_array = np.array([curr_state[0]])
        sampling_times = np.array([0.0])
        for j in range(L + 1):
            if j == L:
                stop_time = final_time
            else:
                stop_time = self.perturbation_times[j]
            times = np.linspace(curr_time, stop_time, num=num_time_samples + 1)
            for k in range(times.size - 1):
                curr_state = self.run_gillespie_ssa(curr_state, times[k + 1] - times[k])
                output_states_array = np.append(output_states_array, [curr_state[0]], axis=0)
                sampling_times = np.append(sampling_times, [times[k + 1]], axis=0)
            if j < L:
                self.param_list[self.param_perturb_index] = self.perturbed_param_vals[j + 1]
            curr_time = stop_time
        return sampling_times, output_states_array

    def generate_sampled_ssa_trajectories(self, stop_time, num_time_samples, num_trajectories=1):
        """
        Create several uniformly sampled SSA Trajectories.
        """
        states_trajectories = np.zeros([num_trajectories, num_time_samples * (len(self.perturbation_times) + 1) + 1])
        for i in range(num_trajectories):
            times, states_trajectories[i, :] = \
                self.generate_sampled_ssa_trajectory(stop_time, num_time_samples)
        return times, states_trajectories

    def stochasticPerturbationAnalysis(self, numTrajectories, final_time=None):
        if not final_time:
            final_time = self.final_time
        times, states_trajectories = self.generate_sampled_ssa_trajectories(final_time, self.numTimeSamples,
                                                                            numTrajectories)
        mean_trajectory = np.mean(states_trajectories, axis=0)
        std_trajectory = np.std(states_trajectories, axis=0) / np.sqrt(numTrajectories)
        self.StochasticMeanOutputTrajectory = [times, mean_trajectory, std_trajectory]

    def visualizeNetwork(self, stochastic=False, title=None):
        if stochastic:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, gridspec_kw={'width_ratios': [1, 0.6, 1.5, 1.5]})
            self.plot_stochasticPerturbationAnalysis(ax4)
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 0.6, 1.5]})
        fig.canvas.manager.set_window_title(title)
        q_str = print_array_nicely(self.q, "q")
        fig.suptitle(q_str + f", $\kappa = {self.kappa:.2f}$, maxRPA score: {self.max_rpa_score:.2f} \%")

        self.drawReactionNetwork(ax1)
        self.writeReactionNetwork(ax2)
        self.plot_deterministic_deterministicPerturbationAnalysis(ax3)
        plt.show()

    def plot_deterministic_deterministicPerturbationAnalysis(self, ax):

        sns.lineplot(x=self.DeterministicOutputTrajectory[0], y=self.DeterministicOutputTrajectory[1], linewidth=1.5,
                     color='g', label="output",
                     ax=ax)
        sns.lineplot(x=self.DeterministicOutputTrajectory[0], y=[self.reference] *
                                                                len(self.DeterministicOutputTrajectory[0]),
                     linewidth=1.5,
                     color='r', label="set-point", ax=ax)
        max_y = ax.get_ylim()[1]
        i = 1
        for perturbation_time in self.perturbation_times:
            ax.vlines(x=perturbation_time, ymin=0, ymax=max_y, linewidth=1.0, linestyle="dashdot", colors='purple',
                      label=f'perturbation' if i == 1 else None)
            i += 1
        ax.lines[0].set_linewidth(2)
        ax.lines[1].set_linestyle("--")
        ax.lines[1].set_linewidth(1)
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x_1(t)$')
        ax.set_title('Deterministic Perturbation Analysis')
        leg = ax.legend(prop=legend_font)
        leg.get_frame().set_edgecolor('black')

    def writeReactionNetwork(self, ax):
        i = 0
        quo = self.numReactions // 10
        for [reaction_reactant, reaction_product] in self.allReactionsList:
            i += 1
            lhs = f"{maxRPA.create_complex_string(reaction_reactant)}"
            if reaction_reactant in self.null_set:
                lhs = f"\emptyset"
            rhs = f"{maxRPA.create_complex_string(reaction_product)}"
            if reaction_product in self.null_set:
                rhs = f"\emptyset"
            s1 = f"${lhs}$"
            s2 = f"${rhs}$"
            param = f"{i}"
            s3 = "$\stackrel{\\theta_{" + param + "}}{\longrightarrow}$"
            ax.text(0.2 - 0.15 * quo + 0.5 * ((i - 1) // 10), 0.95 - 0.1 * (((i - 1) % 10) + 1), s1 + s3 + s2,
                    fontsize=12, ha='left')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        ax.set_title('Network Reactions')

    def drawReactionNetwork(self, ax):
        G = nx.DiGraph()
        edge_list = []
        i = 0
        null_set = [(0, 0), (0,)]
        reaction_no = ["set-point encoding", "sensing"]
        # add the first two reactions
        for [reaction_reactant, reaction_product] in self.first_two_reactions:
            i += 1
            lhs = f"${maxRPA.create_complex_string(reaction_reactant)}$"

            if reaction_reactant in null_set:
                lhs = f"$\emptyset$"
            rhs = f"${maxRPA.create_complex_string(reaction_product)}$"
            if reaction_product in null_set:
                rhs = f"$\emptyset$"
            edge_list.append((lhs, rhs, {'w': reaction_no[i - 1], 'color': 'r'}))

        # add the internal controller reactions
        for [reaction_reactant, reaction_product] in self.internal_controller_reactions_list:
            i += 1
            lhs = f"${maxRPA.create_complex_string(reaction_reactant)}$"
            if reaction_reactant in null_set:
                lhs = f"$\emptyset$"
            rhs = f"${maxRPA.create_complex_string(reaction_product)}$"
            if reaction_product in null_set:
                rhs = f"$\emptyset$"
            reaction_no = str(i)
            edge_list.append((lhs, rhs, {'w': "", 'color': 'k'}))

        # add the interface reactions
        for [reaction_reactant, reaction_product] in self.interface_reactions:
            i += 1
            lhs = f"${maxRPA.create_complex_string(reaction_reactant)}$"
            if reaction_reactant in null_set:
                lhs = f"$\emptyset$"
            rhs = f"${maxRPA.create_complex_string(reaction_product)}$"
            if reaction_product in null_set:
                rhs = f"$\emptyset$"
            reaction_no = str(i)
            # edge_list.append((lhs, rhs, {'w': reaction_no, 'color': 'g'}))
            edge_list.append((lhs, rhs, {'w': "", 'color': 'g'}))

        # add the plant reactions
        for [reaction_reactant, reaction_product] in self.plant_reactions:
            i += 1
            lhs = f"${maxRPA.create_complex_string(reaction_reactant)}$"
            if reaction_reactant in null_set:
                lhs = f"$\emptyset$"
            rhs = f"${maxRPA.create_complex_string(reaction_product)}$"
            if reaction_product in null_set:
                rhs = f"$\emptyset$"
            reaction_no = str(i)
            # edge_list.append((lhs, rhs, {'w': reaction_no, 'color': 'b'}))
            edge_list.append((lhs, rhs, {'w': "", 'color': 'b'}))
        G.add_edges_from(edge_list)
        # pos = nx.kamada_kawai_layout(G)
        # pos = nx.planar_layout(G,scale=5)
        pos = nx.spring_layout(G, k=8)
        # fig, ax = plt.subplots()
        default_node_size = [800] * len(G.nodes())
        j = 0
        for n in G.nodes():
            if len(n) > 6 and n != "$\emptyset$":
                default_node_size[j] = 1500
            j += 1

        nx.draw_networkx_nodes(G, node_size=default_node_size, node_color="grey", pos=pos, ax=ax, alpha=0.3,
                               linewidths=1.0, edgecolors='k')
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)

        curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
        straight_edges = list(set(G.edges()) - set(curved_edges))

        edge_colors = nx.get_edge_attributes(G, 'color')
        straight_edge_colors = [edge_colors[edge] for edge in straight_edges]
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=straight_edges, edge_color=straight_edge_colors)
        arc_rad = 0.25
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}')
        edge_weights = nx.get_edge_attributes(G, 'w')
        curved_edge_labels = {edge: edge_weights[edge] for edge in curved_edges}
        straight_edge_labels = {edge: edge_weights[edge] for edge in straight_edges}
        if curved_edge_labels:
            my_draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=curved_edge_labels, rotate=False, rad=arc_rad)
        if straight_edge_labels:
            nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=straight_edge_labels, rotate=False)
        ax.set_title('Network Design')

    def plot_stochasticPerturbationAnalysis(self, ax):
        [times, mean_trajectory, std_trajectory] = self.StochasticMeanOutputTrajectory
        sns.lineplot(x=times, y=mean_trajectory, color='green', linewidth=1, label="output mean", ax=ax)
        ax.fill_between(times, mean_trajectory - std_trajectory, mean_trajectory + std_trajectory, alpha=0.5,
                        color='grey')
        sns.lineplot(x=times, y=[self.reference] * len(times), linewidth=1.5,
                     color='r', label="set-point", ax=ax)
        max_y = ax.get_ylim()[1]
        i = 1
        for perturbation_time in self.perturbation_times:
            ax.vlines(x=perturbation_time, ymin=0, ymax=max_y, linewidth=1.0, linestyle="dashdot", colors='purple',
                      label=f'perturbation' if i == 1 else None)
            i += 1
        ax.lines[0].set_linewidth(2)
        ax.lines[1].set_linestyle("--")
        ax.lines[1].set_linewidth(1)
        ax.set_xlabel('$t$')
        ax.set_ylabel('$\\mathbb{E}(X_1(t))$')
        ax.set_title('Stochastic Perturbation Analysis')
        leg = ax.legend(prop=legend_font)
        leg.get_frame().set_edgecolor('black')


def my_draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=None,
        label_pos=0.5,
        font_size=10,
        font_color="k",
        font_family="sans-serif",
        font_weight="normal",
        alpha=None,
        bbox=None,
        horizontalalignment="center",
        verticalalignment="center",
        ax=None,
        rotate=True,
        clip_on=True,
        rad=0
):
    """
    This method was taken from :
    https://stackoverflow.com/questions/22785849/drawing-multiple-edges-between-two-nodes-with-networkx

    Draw edge labels.
    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5 * pos_1 + 0.5 * pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0, 1), (-1, 0)])
        ctrl_1 = linear_mid + rad * rotation_matrix @ d_pos
        ctrl_mid_1 = 0.5 * pos_1 + 0.5 * ctrl_1
        ctrl_mid_2 = 0.5 * pos_2 + 0.5 * ctrl_1
        bezier_mid = 0.5 * ctrl_mid_1 + 0.5 * ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items


def print_array_nicely(y, name):
    size_input = y.size
    y = y.reshape(size_input, )
    output = name + "=" + ' ('
    for i in range(y.size - 1):
        output += "{0:.2f}, ".format(y[i])
    output += "{0:.2f})".format(y[y.size - 1])
    return output
