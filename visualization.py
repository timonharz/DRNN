import networkx as nx
import matplotlib.pyplot as plt

def visualize_network(population):
    G = nx.DiGraph()

    for neuron_id, neuron in enumerate(population.neurons):
        neuron_label = f'Neuron {neuron_id}'
        G.add_node(neuron_label, impotance=neuron.importance)

    for neuron_id, neuron in enumerate(population.neurons):
        neuron_label = f'Neuron {neuron_id}'
        for connected_neuron_id in neuron.weights.keys():
            connected_neuron_label = f'Neuron {connected_neuron_id}'
            weight = neuron.weights[connected_neuron_id]
            G.add_edge(neuron_label, connected_neuron_label, weight=weight)

    # Define node colors based on importance
    node_colors = [neuron['importance'] for _, neuron in G.nodes(data=True)]
    node_labels = {node: f'{node}\nImportance: {G.nodes[node]["importance"]:.2f}' for node in G.nodes

    # Draw the DRNN architecture
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.Reds, node_size=500)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, edge_cmap=plt.cm.Blues, width=2, arrows=True)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("DRNN Prototype Architecture")
    plt.axis('off')
    plt.tight_layout()
    plt.show()