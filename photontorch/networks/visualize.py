""" Special Network Plotting Function """

#############
## Imports ##
#############

# Torch
import torch

import re
# Other
import numpy as np

# Relative
from ..components.terms import Detector


##########
## Plot ##
##########


# The plot function to plot the detected power of a network
def plot(network, detected, **kwargs):
    """Plot detected power versus time or wavelength

    Args:
        detected (np.ndarray|Tensor): detected power. Allowed shapes:
            * (#timesteps,)
            * (#timesteps, #detectors)
            * (#timesteps, #detectors, #batches)
            * (#timesteps, #wavelengths)
            * (#timesteps, #wavelengths, #detectors)
            * (#timesteps, #wavelengths, #detectors, #batches)
            * (#wavelengths,)
            * (#wavelengths, #detectors)
            * (#wavelengths, #detectors, #batches)
            the plot function should be smart enough to figure out what to plot.
        **kwargs: keyword arguments given to plt.plot

    Note:
        if #timesteps = #wavelengths, the plotting function will choose #timesteps
        as the first dimension

    """

    import matplotlib.pyplot as plt

    # First we define a helper function
    def plotfunc(x, y, labels, **kwargs):
        """Helper function"""
        plots = plt.plot(x, y, **kwargs)
        if labels is not None:
            for p, l in zip(plots, labels):
                p.set_label(l)
        if labels is not None and len(labels) > 1:
            # Shrink current axis by 10%
            box = plt.gca().get_position()
            plt.gca().set_position([box.x0, box.y0, box.width * 0.85, box.height])
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        return plots

    # Handle y
    y = detected
    if torch.is_tensor(y):
        y = y.detach().cpu()

    if len(y.shape) == 4 and y.shape[0] == 1 and y.shape[1] == 1:
        raise ValueError("cannot plot for a single timestep and a single wavelength")

    y = np.squeeze(np.array(y, "float32"))

    # Handle x
    time_mode = wl_mode = False
    if network.env.num_t == y.shape[0]:
        time_mode = True
        x = network.env.t
    elif network.env.num_wl == y.shape[0]:
        wl_mode = True
        x = network.env.wl
    if not (time_mode or wl_mode):
        raise ValueError("First dimension should be #timesteps or #wavelengths")

    # Handle prefixes
    f = (int(np.log10(max(x)) + 0.5) // 3) * 3 - 3
    prefix = {
        12: "T",
        9: "G",
        6: "M",
        3: "k",
        0: "",
        -3: "m",
        -6: r"$\mu$",
        -9: "n",
        -12: "p",
        -15: "f",
    }[f]
    x = x * 10 ** (-f)

    # Handle labels
    plt.ylabel("Intensity [a.u.]")
    plt.xlabel("Time [%ss]" % prefix if time_mode else "Wavelength [%sm]" % prefix)

    # standard labels:
    detectors = [
        name for name, comp in network.components.items() if isinstance(comp, Detector)
    ]
    wavelengths = ["%inm" % wl for wl in 1e9 * network.env.wl]

    # Plot
    if y.ndim == 1:
        return plotfunc(x, y, None, **kwargs)

    if wl_mode:
        if y.ndim == 2:
            if y.shape[1] == network.num_detectors:
                labels = detectors
            else:
                labels = ["batch %i" % i for i in range(y.shape[1])]
            return plotfunc(x, y, labels, **kwargs)
        elif y.ndim == 3:
            y = y.transpose(0, 2, 1)
            labels = [
                "%i | %s" % (i, det) for i in range(y.shape[1]) for det in detectors
            ]
            return plotfunc(x, y.reshape(network.env.num_wl, -1), labels, **kwargs)
        else:
            raise RuntimeError(
                "When plotting in wavelength mode, the max dim of y should be < 4"
            )

    if time_mode:
        if y.ndim == 2:
            if y.shape[1] == network.env.num_wl:
                labels = wavelengths
            elif y.shape[1] == network.num_detectors:
                labels = detectors
            else:
                labels = ["batch %i" % i for i in range(y.shape[1])]
            return plotfunc(x, y, labels, **kwargs)
        elif y.ndim == 3:
            if y.shape[1] == network.env.num_wl and y.shape[2] == network.num_detectors:
                labels = [
                    "%s | %s" % (wl, det) for wl in wavelengths for det in detectors
                ]
            elif (
                y.shape[1] == network.env.num_wl and y.shape[2] != network.num_detectors
            ):
                y = y.transpose(0, 2, 1)
                labels = [
                    "%i | %s" % (b, wl) for b in range(y.shape[1]) for wl in wavelengths
                ]
            elif y.shape[1] == network.num_detectors:
                y = y.transpose(0, 2, 1)
                labels = [
                    "%i | %s" % (b, det) for b in range(y.shape[1]) for det in detectors
                ]
            return plotfunc(x, y.reshape(network.env.num_t, -1), labels, **kwargs)
        elif y.ndim == 4:
            y = y.transpose(0, 3, 1, 2)
            labels = [
                "%i | %s | %s" % (b, wl, det)
                for b in range(y.shape[1])
                for wl in wavelengths
                for det in detectors
            ]
            return plotfunc(x, y.reshape(network.env.num_t, -1), labels, **kwargs)
        else:
            raise RuntimeError(
                "When plotting in time mode, the max dim of y should be < 5"
            )

    # we should never get here:
    raise ValueError(
        "Could not plot detected array. Are you sure you are you sure the "  # pragma: no cover
        "current simulation environment corresponds to the environment for "
        "which the detected tensor was calculated?"
    )
    
import networkx as nx
def graph(network,layout = nx.spring_layout, draw=True):
    """Create a graph visualization of the network, starting from each component up.

    Args:
        network: The network to visualize.
        draw (bool): Draw the graph with matplotlib.

    Returns:
        nx.MultiGraph: The networkx MultiGraph object representing the network.
    """
    
    import matplotlib.pyplot as plt


    def add_nodes_recursively(G,component,component_name,Nodes,Connections,parents=list()): 
        
        if component and hasattr(component, 'components'):

            cop1 = list(parents)
            cop1.append(component.name)
            for child in component.components:

                add_nodes_recursively(G, component.components.get(child),child,Nodes,Connections, cop1)  # Recursively add nodes
                
                
                
            for connection in component.connections:
                con = connection.split(":")
                if (len(con)==3):
                    Connections.append(((cop1+[con[0]],con[1]),(cop1,con[2])))
                else:
                    Connections.append(((cop1+[con[0]],con[1]),(cop1+[con[2]],con[3])))

                    
        else:
            
            if component not in Nodes:
                
                
                copy = list(parents)
                copy.append(component_name)
                Nodes.append(copy)
                
                
    def fix_connections(connections,nodes):
        connections_dict = {}

        # Construct the dictionary of second elements and their corresponding first elements
        for pair in connections:
            first,second = pair
            connections_dict[tuple((tuple(first[0]),first[1]))]=tuple((tuple(second[0]),second[1]))
        output=list()
        # Iterate through the connections and fix any duplicates
        for first in reversed(connections_dict.keys()):
            first = tuple((tuple(first[0]),first[1]))
            second = connections_dict.get(first)
            second = tuple((tuple(second[0]),second[1]))
            flag = True
            
            for i,value in enumerate(output):
                
                if (first == value[0]):
                    output[i][0] = second
                    flag = False
                elif (first == value[1]):
                    output[i][1] = second
                    flag = False
                elif (second == value[0]):
                    output[i][0] = first
                    flag = False
                elif (second == value[1]):
                    output[i][1] = first
                    flag = False

                
            if(flag):
                output.append([first,second])
            
            flag = True
        
        node_tuples = [tuple(node) for node in Nodes]
        if ('recknxn_terminated', 'recknxn', 'wg') in node_tuples:
            node_tuples.remove(('recknxn_terminated', 'recknxn', 'wg'))
        
        output_a = [pair for pair in output if pair[0][0] in node_tuples and pair[1][0] in node_tuples]
        dels = [pair for pair in output if pair not in output_a]
        
        middle_index = len(dels) // 2
        first_set = dels[:middle_index]
        second_set = dels[middle_index:]

        combined_list = [(first_set[i][0], second_set[i][0]) for i in range(len(first_set))]
        output_a = output_a+combined_list
        
        return output_a


    # Create graph
    G = nx.MultiGraph()
    Nodes=list()
    Connections=list()
    add_nodes_recursively(G, network,network.name,Nodes,Connections)    
    Connections = fix_connections(Connections,Nodes)
    node_names = ['_'.join(sublist) for sublist in Nodes]
    if 'recknxn_terminated_recknxn_wg' in node_names:
        node_names.remove('recknxn_terminated_recknxn_wg')


    G.add_nodes_from(node_names)

    connection_names = [[pair[0] for pair in sublist] for sublist in Connections]
    connection_names_2 = [['_'.join(sublist[0]),'_'.join(sublist[1])] for sublist in connection_names]
    G.add_edges_from(connection_names_2)

    if draw:
        pos = layout(G)
        for node in pos:
            pos[node]=np.sign(pos[node])*np.power(np.abs(pos[node]),1/4)*5
        plt.figure(figsize=(10, 10))
        _draw_nodes(G, Nodes, pos)
        _draw_edges(G, pos)
        
        plt.gca().set_axis_off()
        plt.draw()
        plt.show()

    return G


import numpy as np

def _draw_nodes(G, Nodes, pos):
    """helper function: draw the nodes of a networkx graph of a photontorch network

    Args:
        G (graph): networkx graph to draw
        components (list): list of Photontorch components in the graph.
        pos (list): list of positions to draw the graph nodes on.

    """
    import matplotlib.pyplot as plt

    nodelist = list(G)
    node_size = 20
    node_color = "r"
    node_shape = "o"
    xy = np.asarray([pos[v] for v in nodelist])

    pattern_mzi = re.compile(r"mzi(\d+)$")
    pattern_wg = re.compile(r"wg(\d+)?$")
    
    for (x, y), node, comp in zip(xy, nodelist, Nodes):
                
        node_color = "y"
        if (pattern_mzi.search(node)):
            node_color = "r"
        if (pattern_wg.search(node)):
            node_color = "k"
        
        plt.scatter(x, y, s=node_size, c=node_color, marker=node_shape, zorder=2)
        
        text = plt.text(
            x,
            y+.1,
            node,
            zorder=2,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=8,
        )


def _draw_edges(G, pos):
    """helper function: draw the edges of a networkx graph of a photontorch network

    Args:
        G (graph): networkx graph to draw
        pos (list): list of positions to draw the edges between.

    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import FancyArrowPatch

    edge_color = "k"
    style = "solid"
    edgelist = list(G.edges())
    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])
    for (x1, y1), (x2, y2) in edge_pos:
        r = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        a = np.arctan2((y2 - y1), (x2 - x1))
        #a += (2 * np.random.rand() - 1) * np.pi / 4
        plt.plot(
            [x1, x1 + 0.5 * r * np.cos(a), x2],
            [y1, y1 + 0.5 * r * np.sin(a), y2],
            lw=1.5,
            ls="-",
            color="k",
            zorder=1,
        )
    
