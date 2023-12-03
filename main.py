from drnn import DRNN
from population import Population

def main():
    input_size = 10
    output_size = 2
    population_size = 5
    max_generations = 10
    initial_neurons = 10  # Specify the initial number of neurons
    layer_structure = [8, 6, 4]  # Specify the layer structure of the DRNN
    importance_threshold = 0.5
    add_neuron_prob = 0.2
    remove_neuron_prob = 0.1

    drnn = DRNN(population_size, input_size, output_size, initial_neurons, layer_structure,
                importance_threshold, add_neuron_prob, remove_neuron_prob)

    population = Population(population_size, input_size, output_size, initial_neurons, layer_structure)

    for generation in range(max_generations):
        print(f"Generation: {generation + 1}")
        # Perform reproduction and dynamic modification
        population.reproduce(importance_threshold=0.5,
                             add_neuron_prob=0.2,
                             remove_neuron_prob=0.1)
        population.dynamic_modification(importance_threshold=0.5,
                                        add_neuron_prob=0.2,
                                        remove_neuron_prob=0.1)

        # Evaluate and update fitness of each network in the population
        population.evaluate_fitness()

        # Print the fitness scores of the population
        population.print_fitness_scores()

        # Select the best networks for the next generation
        population.selection()
    best_network = population.get_best_network()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
