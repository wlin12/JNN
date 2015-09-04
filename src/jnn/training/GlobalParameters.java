package jnn.training;

import util.IOUtils;

public class GlobalParameters {

	public static String cudaLibraryDir = "/lib/";

	public static String nd4jLibraryDir = "/tmp/";

	public static String nd4jPropertiesFile = "src/resources/nd4j.properties";
	
	public static String nd4jResourceDir = "src/resources/";

	public static double learningRateDefault = 0.05;
	
	public static boolean useAdagradDefault = false;

	public static boolean useMomentumDefault = false;

	public static boolean useAdadeltaDefault = false;

	public static boolean useAdagradL1Default = false;
	
	public static double l2regularizerLambdaDefault = 0;
	
	public static double noiseDevDefault = 0.1;
	
	public static boolean addNoiseDefault = false;

	public static boolean sparseDefault = false;	

	public static int commitMethodDefault = 1;

	public static double maxVal = 10000;
	
	public static double maxError = 1;

	public static double adagradL1LambdaDefault=0;

	public static boolean fastUpdate = false;
	
	public static double momentumDefault = 0.95;

	public static double adadeltaMomentumDefault = 0.95;

	public static double adadeltaEpsDefault = 0.000001;

	public static int initializationType = 0; // 0 -> uniform 0-0.1, 1 -> based on fanin fanout;

	public static void setUpdateMethod(String optionValue) {
		useMomentumDefault = false;
		useAdagradDefault = false;
		useAdadeltaDefault = false;
		if(optionValue.equals("momentum")){
			useMomentumDefault = true;
		}
		else if(optionValue.equals("adagrad")){
			useAdagradDefault = true;
		}
		else if(optionValue.equals("adadelta")){
			useAdadeltaDefault = true;
		}
	}

	public static void setND4JResourceDir(String dir) {
		nd4jLibraryDir = dir+"/lib";
		IOUtils.mkdir(nd4jLibraryDir);
		nd4jPropertiesFile = dir+"/nd4j.properties";
		nd4jResourceDir = dir;
		cudaLibraryDir = dir;
	}
	
}
