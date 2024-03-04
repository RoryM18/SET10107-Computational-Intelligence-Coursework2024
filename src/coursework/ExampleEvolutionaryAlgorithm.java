package coursework;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

import model.Fitness;
import model.Individual;
import model.LunarParameters.DataSet;
import model.NeuralNetwork;

/**
 * Implements a basic Evolutionary Algorithm to train a Neural Network
 * 
 * You Can Use This Class to implement your EA or implement your own class that extends {@link NeuralNetwork} 
 * 
 */
public class ExampleEvolutionaryAlgorithm extends NeuralNetwork {
	
	private double mutationRate = Parameters.mutateRate;
    private int noImprovementCounter = 0;
    
    double previousBestFitness = Double.MAX_VALUE; // Initialize with a large value
    
    double[][] pheromones = new double[Parameters.popSize][Parameters.numGenes];
	
	/**
	 * The Main Evolutionary Loop
	 */
	@Override
	public void run() {		
		//Initialise a population of Individuals with random weights
		population = initialise();

		//Record a copy of the best Individual in the population
		best = getBest();
		System.out.println("Best From Initialisation " + best);

		/**
		 * main EA processing loop
		 */		
				
		while (evaluations < Parameters.maxEvaluations) {

			/**
			 * this is a skeleton EA - you need to add the methods.
			 * You can also change the EA if you want 
			 * You must set the best Individual at the end of a run
			 * 
			 */
			
			
			
			// Select 2 Individuals from the current population. Tournament selection 
			Individual parent1; 
			Individual parent2;
			switch(Parameters.selection) 
			{
			case "roulette":
				parent1 = rouletteWheelSelection(population); 
				parent2 = rouletteWheelSelection(population);
				break;
			case "tournament":
				parent1 = tournamentSelection(population, 10);
				parent2 = tournamentSelection(population, 10);
				break;
			case "elitist":
				Collections.sort(population);
				parent1 = population.get(0);
				parent2 = population.get(1);
				break;
			case "ant":
				parent1 = pheromoneBasedSelection(population);
				parent2 = pheromoneBasedSelection(population);
				
				updatePheromones(population);
				
				evaporatePhermones();
				break;
			default:
				parent1 = randomSelection();
				parent2 = randomSelection();
			}
			

			// Generates children 
			ArrayList<Individual> children;
			switch(Parameters.reproduction) 
			{
			case "twoPoint":
				children = twoPointCrossover(parent1, parent2);
				break;
			case "uniform":
				children = uniformCrossover(parent1, parent2);
				break;
			case "linear":
				children = linearCrossover(parent1, parent2, 0.5);
				break;
			default:
				children = reproduce(parent1, parent2);
			}			
			
			//mutate the offspring
			switch(Parameters.mutation) 
			{
			case "swap":
				swapMutation(children);
				break;
			case "flipBit":
				flipBitMutate(children);
				break;
			case "inversion":
				inversionMutation(children);
				break;
			case "scramble":
				scrambleMutation(children);
				break;
			default: 
				mutate(children);
			}
			
			// Evaluate the children
			evaluateIndividuals(children);
			
			//Adjust Parameters
			//adjustParameters();

			// Replace children in population 
			switch(Parameters.replacement) 
			{
			case "best":
				replaceBest(children, parent1, parent2);
				break;
			case "generational":
				replaceGenerational(children);
				break;
			case "steadyState":
				replaceSteadyState(children);
				break;
			case "bestRanked":
				replaceBestRankBased(children);
				break;
			case "restrictedTSReplacement":
				restrictedTSReplacement(children);
				break;
			default:
				replace(children);
			}

			// check to see if the best has improved
			best = getBest();
			
			// check to see if the best has improved
	        double currentBestFitness = best.fitness;
	        
	        if (currentBestFitness < previousBestFitness) 
	        {
	            // There's an improvement, reset the noImprovementCounter
	            noImprovementCounter = 0;
	            previousBestFitness = currentBestFitness;
	        } 
	        else 
	        {
	            // No improvement, increment the counter
	            noImprovementCounter++;
	        }
		
			
			// Implemented in NN class. 
			outputStats();
			
			//Increment number of completed generations			
		}

		//save the trained network to disk
		saveNeuralNetwork();
	}
	
	private void adjustParameters() 
	{
		//Implement logic to dynamically change parameters such as mutation rate and crossover rate
		//Based on certain conditions 
		
		// Example: Adjust mutation rate based on some condition
	    if (noImprovementCounter >= 50) 
	    {
	        mutationRate += Parameters.mutationDecreaseFactor;
	        noImprovementCounter = 0; // Reset the counter
	        System.out.println("Mutation rate adjusted to: " + mutationRate);
	    }
	    else if(noImprovementCounter == 0)
	    {
	    	mutationRate += Parameters.mutationIncreaseFactor;
	    	System.out.println("Mutation rate adjusted to: " + mutationRate);
	    }
    }

	/**
	 * Sets the fitness of the individuals passed as parameters (whole population)
	 * 
	 */
	private void evaluateIndividuals(ArrayList<Individual> individuals) {
		for (Individual individual : individuals) {
			individual.fitness = Fitness.evaluate(individual, this);
		}
	}


	/**
	 * Returns a copy of the best individual in the population
	 * 
	 */
	private Individual getBest() {
		best = null;;
		for (Individual individual : population) {
			if (best == null) {
				best = individual.copy();
			} else if (individual.fitness < best.fitness) {
				best = individual.copy();
			}
		}
		return best;
	}

	/**
	 * Generates a randomly initialised population
	 * 
	 */
	private ArrayList<Individual> initialise() {
		population = new ArrayList<>();
		for (int i = 0; i < Parameters.popSize; ++i) {
			//chromosome weights are initialised randomly in the constructor
			Individual individual = new Individual();
			population.add(individual);
		}
		evaluateIndividuals(population);
		return population;
	}

	/**
	 * Selection --
	 * 
	 * NEEDS REPLACED with proper selection this just returns a copy of a random
	 * member of the population
	 */
	private Individual randomSelection() 
	{		
		Individual parent = population.get(Parameters.random.nextInt(Parameters.popSize));
		return parent.copy();
	}
	
	private Individual rouletteWheelSelection(ArrayList<Individual> population)
	{
		//Calculate total fitness
		double totalFitness = 0;
		for(Individual individual : population)
		{
			totalFitness += individual.fitness;
		}
		
		//Generate a random value between 0 and totalFitness
		double randValue =Math.random() * totalFitness;
		
		//Perform Roulette Wheel selection
		double cumulativeFitness = 0;
		for(Individual individual : population)
		{
			cumulativeFitness += individual.fitness;
			if(cumulativeFitness >= randValue)
			{
				// Found the selected individual 
				return individual.copy();
			}
		}
		
		//This should not happen under normal circumstances but if it does return the last individual
		return population.get(population.size() - 1).copy();
	}
	
	private Individual tournamentSelection(ArrayList<Individual> population, int tournamentSize) {
	    // Ensure the tournament size is within the bounds of the population size
	    tournamentSize = Math.min(tournamentSize, population.size());

	    // Randomly select individuals for the tournament
	    ArrayList<Individual> tournament = new ArrayList<>();
	    for (int i = 0; i < tournamentSize; i++) {
	        int randomIndex = Parameters.random.nextInt(population.size());
	        tournament.add(population.get(randomIndex));
	    }

	    // Find the individual with the best fitness in the tournament
	    Individual bestIndividual = tournament.get(0);
	    for (Individual individual : tournament) {
	        if (individual.fitness < bestIndividual.fitness) {
	            bestIndividual = individual;
	        }
	    }

	    return bestIndividual.copy(); // Assuming you have a copy method
	}

	/**
	 * Crossover / Reproduction
	 * 
	 * NEEDS REPLACED with proper method this code just returns exact copies of the
	 * parents. 
	 */
	private ArrayList<Individual> reproduce(Individual parent1, Individual parent2) {
		ArrayList<Individual> children = new ArrayList<>();
		children.add(parent1.copy());
		children.add(parent2.copy());		
		return children;
	} 
	
	private ArrayList<Individual> uniformCrossover(Individual parent1, Individual parent2)
	{
		ArrayList<Individual> children = new ArrayList<>();
		
		for (int i = 0; i < parent1.chromosome.length; i++) {
	        Individual child = new Individual();

	        if (Math.random() < 0.5) {
	            // Take the gene from parent1
	            child.chromosome[i] = parent1.chromosome[i];
	        } else {
	            // Take the gene from parent2
	            child.chromosome[i] = parent2.chromosome[i];
	        }

	        children.add(child);
	    }

	    return children;
	}
	
	private ArrayList<Individual> twoPointCrossover(Individual parent1, Individual parent2)
	{
		ArrayList<Individual> children = new ArrayList<>();
		
		//Choose two random crossover points
		int crossoverPoint1 = Parameters.random.nextInt(parent1.chromosome.length);
		int crossoverPoint2 = Parameters.random.nextInt(parent1.chromosome.length);
		
		//Ensure crossoverPoint1 is before crossoverPoiint2
		if(crossoverPoint1 > crossoverPoint2)
		{
			int temp = crossoverPoint1;
			crossoverPoint1 = crossoverPoint2;
			crossoverPoint2 = temp;
		}
		
		//Create children using two-point crossover
		Individual child1 = new Individual();
		Individual child2 = new Individual();
		
		//Copy genes from parents up to crossoverPoint1
		for(int i = 0; i < crossoverPoint1; i++)
		{
			int temp = crossoverPoint1;
			crossoverPoint1 = crossoverPoint2;
			crossoverPoint2 = temp;
		}
		
		//Copy genes from parent 2 between crossoverpoint1 and crossoverPoint2
		for(int i = crossoverPoint1; i < crossoverPoint2; i++)
		{
			child1.chromosome[i] = parent2.chromosome[i];
			child2.chromosome[i] = parent1.chromosome[i];
		}
		
		//Copy genes from parent 2 between crossoverpoint1 and crossoverPoint2
		for(int i = crossoverPoint2; i < parent1.chromosome.length; i++)
		{
			child1.chromosome[i] = parent1.chromosome[i];
			child2.chromosome[i] = parent2.chromosome[i];
		}
		
		//Add the children to the list
		children.add(child1);
		children.add(child2);
		
		return children;
	}
	
	private ArrayList<Individual> linearCrossover(Individual parent1, Individual parent2, double alpha)
	{
		ArrayList<Individual> children = new ArrayList<>();
		
		Individual child1 = new Individual();
		Individual child2 = new Individual();
		
		for(int i = 0; i < parent1.chromosome.length; i ++)
		{
			//Linear Crossover Formula 
			child1.chromosome[i] = alpha * parent1.chromosome[i] + (1 - alpha) * parent2.chromosome[i];
			child2.chromosome[i] = alpha * parent2.chromosome[i] + (1 - alpha) * parent1.chromosome[i];
			
			//Ensure values stay within the specified range
			child1.chromosome[i] = Math.max(Parameters.minGene, Math.min(Parameters.maxGene, child1.chromosome[i]));
			child2.chromosome[i] = Math.max(Parameters.minGene, Math.min(Parameters.maxGene, child2.chromosome[i]));
		}
		
		children.add(parent1);
		children.add(parent2);
		
		return children;
	}
	
	/**
	 * Mutation
	 * 
	 * 
	 */
	private void mutate(ArrayList<Individual> individuals) {		
		for(Individual individual : individuals) {
			for (int i = 0; i < individual.chromosome.length; i++) {
				if (Parameters.random.nextDouble() < Parameters.mutateRate) {
					if (Parameters.random.nextBoolean()) {
						individual.chromosome[i] += (Parameters.mutateChange);
					} else {
						individual.chromosome[i] -= (Parameters.mutateChange);
					}
				}
			}
		}		
	}
	
	private void swapMutation(ArrayList<Individual> individuals)
	{
		for(Individual individual : individuals)
		{
			for(int i = 0; i < individual.chromosome.length; i++)
			{
				int chromosomeLength = individual.chromosome.length;
				
				int gene1 = Parameters.random.nextInt(chromosomeLength); 
				int gene2 = Parameters.random.nextInt(chromosomeLength);
				
				//Swap position of genes
				swapGenes(individual.chromosome, gene1, gene2);
			}
		}
	}
	
	private void swapGenes(double[] array, int gene1, int gene2)
	{
		//Swap elements at gene1 and gene2
		double temp = array[gene1];
		array[gene1] = array[gene2];
		array[gene2] = temp;
	}
	
	private void flipBitMutate(ArrayList<Individual> individualas)
	{
		for(Individual individual : individualas)
		{
			for(int i = 0; i < individual.chromosome.length; i++)
			{
				if(Parameters.random.nextDouble() < Parameters.mutateRate)
				{
					//Flip the value of the randomly selected bit
					individual.chromosome[i] = 1 - individual.chromosome[i];
				}
			}
		}
	}
	
	private void inversionMutation(ArrayList<Individual> individuals)
	{
		for(Individual individual : individuals)
		{
			if(Parameters.random.nextDouble() < Parameters.mutateRate)
			{
				int chromosomeLength = individual.chromosome.length;
				
				//Select a random subset of genes to invert
				int start = Parameters.random.nextInt(chromosomeLength);
				int end = Parameters.random.nextInt(chromosomeLength - start) + start;
				
				//Perform theinversion mutation
				invertSubarray(individual.chromosome, start, end);
			}
		}
	}
	
	private void invertSubarray(double[] array, int start, int end)
	{
		while(start < end)
		{
			//Swap elements at start and end
			double temp = array[start];
			array[start] = array[end];
			array[end] = temp;
			
			//Move indices towards the center
			start++;
			end--;
		}
	}
	
	private void scrambleMutation(ArrayList<Individual> individuals)
	{
		for(Individual individual : individuals)
		{
			if(Parameters.random.nextDouble() < Parameters.mutateRate)
			{
				int chromosomeLength = individual.chromosome.length;
				
				//Select a random subset of gene to scramble
				int start = Parameters.random.nextInt(chromosomeLength);
				int end = Parameters.random.nextInt(chromosomeLength - start) + start;
				
				// Perform the scramble mutation
				scrambleSubarray(individual.chromosome, start, end);
			}
		}
	}
	
	private void scrambleSubarray(double[] array, int start, int end)
	{
		int length = end - start + 1;
		double[] subarray = new double[length];
		
		//Copy the selected subset of genes into a tempoaray subarray
		System.arraycopy(array, start, subarray, 0, length);
		
		//Shuffle the elements in the temporary subarray
		shuffleArray(subarray);
		
		//Copy the shuffled subarray back into the original array
		System.arraycopy(subarray, 0, array, start, length);
	}
	
	private void shuffleArray(double[] array)
	{
		int index;
		double temp;
		for(int i = array.length - 1; i > 0; i--)
		{
			index = Parameters.random.nextInt(i + 1);
			//Swap elements at i and index
			temp = array[i];
			array[i] = array[index];
			array[index] = temp;
		}
	}

	/**
	 * 
	 * Replaces the worst member of the population 
	 * (regardless of fitness)
	 * 
	 */
	private void replace(ArrayList<Individual> individuals) {
	    int worstIndex = getWorstIndex();
	    
	    for (Individual individual : individuals) {
	        population.set(worstIndex, individual);
	    }
	}
	
	//Replaces parents only if children are fitter: 
	private void replaceBest(ArrayList<Individual> children, Individual parent1, Individual parent2) 
	{
		children.add(parent2);
		children.add(parent1);
		population.remove(parent1);
		population.remove(parent2);
		
		Collections.sort(children);
		Individual best = children.get(0);
		Individual secondBest = children.get(1);
		
		population.add(best);
		population.add(secondBest);
	}

	private void replaceGenerational(ArrayList<Individual> children)
	{
		population.clear();
		population.addAll(children);
	}
	
	private void replaceSteadyState(ArrayList<Individual> children)
	{
		for(Individual child : children)
		{
			int worstIndex = getWorstIndex();
			population.set(worstIndex, child);
		}
	}
	
	private void replaceBestRankBased(ArrayList<Individual> children)
	{
		Collections.sort(population, new Comparator<Individual>()
		{
			@Override
			public int compare(Individual o1, Individual o2)
			{
				return Double.compare(o1.fitness, o2.fitness);
			}
		});
		
		Individual bestChild = Collections.min(children, new Comparator<Individual>() {
	        @Override
	        public int compare(Individual o1, Individual o2) {
	            return Double.compare(o1.fitness, o2.fitness);
	        }
	    });
		
		//Replace worst individual with the best-ranked child 
		population.set(getWorstIndex(), bestChild);
	}
	
	// Restricted tournament Replacement
	private void restrictedTSReplacement(ArrayList<Individual> children)
	{
		for(Individual child : children)
		{
			//Make a tournament of random Individuals
			ArrayList<Individual> tournament = new ArrayList<Individual>();
			for(int i = 0; i < Parameters.tournamentSize; i++)
			{
				tournament.add(population.get(Parameters.random.nextInt(Parameters.popSize)));
			}
			
			//Set first tournament member as most similar
			Individual mostSimilar = tournament.get(0);
			double minDist = getDistance(child, mostSimilar);
			
			//Compare all tournament members to find the most similar one
			for(int i = 0; i < Parameters.tournamentSize; i++)
			{
				if(getDistance(tournament.get(i), child) < minDist)
				{
					mostSimilar = tournament.get(i);
					minDist = getDistance(tournament.get(i), child);
				}
			}
			
			//If the child is fitter than most similar members, replace it
			if(child.fitness < mostSimilar.fitness)
			{
				population.set(population.indexOf(mostSimilar), child);
			}
		}
	}
	
	//return the distance between individuals by adding absolute difference of chromosome values
		private double getDistance(Individual indi1, Individual indi2)
		{
			double distance = 0;
			for(int i = 0; i < indi1.chromosome.length; i++)
			{
				distance += Math.abs(indi1.chromosome[i] - indi2.chromosome[i]);
			}
			return distance;
		}
		
	/**
	 * Ant Colony optimisation experimentation
	 * @return
	 */
	
	
	private void updatePheromones(ArrayList<Individual> individuals) {
	    for (int i = 0; i < individuals.size(); i++) {
	        Individual individual = individuals.get(i);
	        // Update pheromones based on individual's fitness
	        for (int j = 0; j < Parameters.numGenes; j++) {
	            pheromones[i][j] += individual.fitness;
	        }
	    }
	}
	
	private Individual pheromoneBasedSelection(ArrayList<Individual> population)
	{
		double[] probabilities = calculateSelectionProbability(population);
	    double randValue = Math.random();
	    double cumulativeProbability = 0.0;

	    for (int i = 0; i < Parameters.popSize; i++) 
	    {
	        cumulativeProbability += probabilities[i];
	        if (randValue <= cumulativeProbability) 
	        {
	            return population.get(i).copy(); // Assuming you have a copy method in your Individual class
	        }
	    }

	    // This should not happen under normal circumstances, but if it does, return the last individual
	    return population.get(Parameters.popSize - 1).copy();
	}
	
	private double calculateTotalPheromones(ArrayList<Individual> population) {
	    double total = 0.0;
	    
	    for (int i = 0; i < population.size(); i++) {
	        Individual individual = population.get(i);
	        for (int j = 0; j < Parameters.numGenes; j++) {
	            total += pheromones[i][j];
	        }
	    }
	    return total;
	}
	
	private void evaporatePhermones()
	{
		for(int i = 0; i < Parameters.popSize; i++)
		{
			for(int j = 0; j < Parameters.numGenes; j++)
			{
				pheromones[i][j] *= (1.0 - Parameters.pheremoneEvaperationRate);
			}
		}
	}
	
	private double[] calculateSelectionProbability(ArrayList<Individual> population) {
	    double[] probabilities = new double[Parameters.popSize];
	    double totalPheromone = calculateTotalPheromones(population);

	    for (int i = 0; i < Parameters.popSize; i++) {
	        double individualPheromone = 0.0;
	        for (int j = 0; j < Parameters.numGenes; j++) {
	            individualPheromone += pheromones[i][j];  // Use the index directly
	        }

	        // Normalize the pheromone value to calculate probability
	        probabilities[i] = individualPheromone / totalPheromone;
	    }

	    return probabilities;
	}
	
	
	/**
	 * Returns the index of the worst member of the population
	 * @return
	 */
	private int getWorstIndex() {
		Individual worst = null;
		int idx = -1;
		for (int i = 0; i < population.size(); i++) {
			Individual individual = population.get(i);
			if (worst == null) {
				worst = individual;
				idx = i;
			} else if (individual.fitness > worst.fitness) {
				worst = individual;
				idx = i; 
			}
		}
		return idx;
	}	

	@Override
	public double activationFunction(double x) {
		if (x < -20.0) {
			return -1.0;
		} else if (x > 20.0) {
			return 1.0;
		}
		return Math.tanh(x);
	}
}
