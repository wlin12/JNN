package util;

import java.util.HashMap;
import java.util.Random;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

public class RandomUtils {

	static Random randGen = new Random(); 
	static HashMap<Integer,RandomGenerator> randomGenPerIt = new HashMap<Integer, RandomGenerator>(); 

	public static void initializeRandomMatrix(double[][] a, double minus, double norm){
		for(int i = 0; i < a.length; i++){
			for(int j = 0; j < a[i].length; j++){
				a[i][j] = (randGen.nextDouble()-minus)/norm;
			}
		}
	}
	
	public static void initializeRandomMatrix(double[][] a, double min, double max, double norm){
		double mult = max - min;
		for(int i = 0; i < a.length; i++){
			for(int j = 0; j < a[i].length; j++){
				a[i][j] = (randGen.nextDouble()*mult + min)/norm;
			}
		}
	}

	public static void initializeRandomArray(double[] a, double minus, double norm){
		for(int i = 0; i < a.length; i++){
			a[i] = (randGen.nextDouble()-minus)/norm;
		}
	}

	public static void initializeRandomArray(double[] a, double min, double max, double norm){
		double mult = max - min;
		for(int i = 0; i < a.length; i++){
			a[i] = (randGen.nextDouble()*mult + min)/norm;
		}		
	}
	
	public static double initializeRandomNumber(double min, double max, double norm){
		double mult = max - min;
		return (randGen.nextDouble()*mult + min)/norm;
	}
	
	public static int initializeRandomInteger(double min, double max){
		double mult = max - min + 1;
		return (int)(randGen.nextDouble()*mult + min);
	}
	
	public static void shuffleArray(int[] ar)
	  {
	    Random rnd = new Random();
	    for (int i = ar.length - 1; i > 0; i--)
	    {
	      int index = rnd.nextInt(i + 1);
	      // Simple swap
	      int a = ar[index];
	      ar[index] = ar[i];
	      ar[i] = a;
	    }
	  }
	
	public static RandomGenerator getRandomGenerator(int id){
		if(!randomGenPerIt.containsKey(id)){
			initRandomGenerator(id, 123);
		}
		return randomGenPerIt.get(id);
	}
	
	public synchronized static void initRandomGenerator(int id, int seed){
		MersenneTwister rng = new MersenneTwister(seed);			
		randomGenPerIt.put(id, rng);		
	}
	
	public static void main(String[] args){
		double[] a = new double[10];
		initializeRandomArray(a, -4, 4, 1);
		PrintUtils.printDoubleArray("vals ", a,false);
	}
}
