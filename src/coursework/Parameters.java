package coursework;

import java.lang.reflect.Field;
import java.util.Random;
import model.LunarParameters;
import model.NeuralNetwork;
import model.LunarParameters.DataSet;

public class Parameters {

	/**
	 * These parameter values can be changed You may add other Parameters as
	 * required to this class
	 * 
	 */
	
	//selection: roulette, tournament, elitist, random
	//crossover: twoPoint, uniform, random
	//mutation: mutate, swapMutation, flipBitMutation, inversionMutation, scrambleMutation
	//replacement: replace, replaceBest, replaceGenerational, replaceSteadyState, replaceBestRanked, restrictedTSReplacement
	
	public static String selection = "ant"; 
	public static String reproduction = "linear";
	public static String mutation ="flipBit";
	public static String replacement = "restrictedTSReplacement";
	
	private static int numHidden = 5;
	public static int numGenes = calculateNumGenes();
	public static double minGene = -1.0; // specifies minimum and maximum weight values
	public static double maxGene = +1.0;

	public static int popSize = 40;
	public static int maxEvaluations = 20000;
	public static int thresholdGenerations = 100;
	public static int generationInterval = 100;

	// Parameters for mutation
	// Rate = probability of changing a gene
	// Change = the maximum +/- adjustment to the gene value
	public static double mutateRate = 0.02; // 0.01 mutation rate for mutation operator
	public static double mutateChange = 0.1; // delta change for mutation operator
	public static double mutationIncreaseFactor = 0.001;
	public static double mutationDecreaseFactor = -0.001;
	public static double pheremoneEvaperationRate = 0.04;
	
	// Tournament Size
	public static int tournamentSize = 10;

	// Random number generator used throughout the application
	public static long seed = System.currentTimeMillis();
	public static Random random = new Random(seed);

	// set the NeuralNetwork class here to use your code from the GUI
	public static Class neuralNetworkClass = ExampleEvolutionaryAlgorithm.class;

	/**
	 * Do not change any methods that appear below here.
	 * 
	 */

	public static int getNumGenes() {
		return numGenes;
	}

	private static int calculateNumGenes() {
		int num = (NeuralNetwork.numInput * numHidden) + (numHidden * NeuralNetwork.numOutput) + numHidden
				+ NeuralNetwork.numOutput;
		return num;
	}

	public static int getNumHidden() {
		return numHidden;
	}

	public static void setHidden(int nHidden) {
		numHidden = nHidden;
		numGenes = calculateNumGenes();
	}

	public static String printParams() {
		String str = "";
		for (Field field : Parameters.class.getDeclaredFields()) {
			String name = field.getName();
			Object val = null;
			try {
				val = field.get(null);
			} catch (IllegalArgumentException | IllegalAccessException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			str += name + " \t" + val + "\r\n";

		}
		return str;
	}

	public static void setDataSet(DataSet dataSet) {
		LunarParameters.setDataSet(dataSet);
	}

	public static DataSet getDataSet() {
		return LunarParameters.getDataSet();
	}

	public static void main(String[] args) {
		printParams();
	}
}
