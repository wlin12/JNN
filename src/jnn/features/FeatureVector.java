package jnn.features;

import java.io.PrintStream;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Scanner;

import util.PrintUtils;
import util.RandomUtils;
import util.SerializeUtils;

public class FeatureVector {


	public static double learningRateDefault = 0.05;
	
	public static boolean useAdagradDefault = false;

	public static boolean useMomentumDefault = false;

	public static boolean useAdadeltaDefault = false;

	public static boolean useAdagradL1Default = false;
	
	public static double l2regularizerLambdaDefault = 0.005;
	
	public static double noiseDevDefault = 0.01;
	
	public static boolean addNoiseDefault = true;

	public static boolean sparseDefault = false;	

	public static int commitMethodDefault = 1;

	public static double maxVal = 10000;
	
	public static double maxError = 1;

	public static double adagradL1LambdaDefault=0.00025;

	public static boolean fastUpdate = false;
	
	public static double momentumDefault = 0.95;

	public static double adadeltaMomentumDefault = 0.95;

	public static double adadeltaEpsDefault = 0.000001;

	public static int initializationType = 0; // 0 -> uniform 0-0.1, 1 -> based on fanin fanout;
	
	
}
