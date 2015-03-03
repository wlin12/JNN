package util;

import java.io.PrintStream;
import java.util.List;

public class PrintUtils {

	public static boolean disablePrints = true;

	public static void printString(String str){
		if(!disablePrints){
			System.err.println(str);
		}
	}

	public static void printDouble(String tag, double val, boolean debug){
		if(!disablePrints || !debug){
			System.err.println(tag + " " + val);
		}
	}

	public static void printDouble(String tag, double val){
		printDouble(tag, val, false);
	}

	public static void printDoubleArray(String tag, double[] vals, boolean debug){
		if(!disablePrints || !debug){
			System.err.print(tag + "[len:" + vals.length + "] ");
			for(int i = 0; i < vals.length; i++){
				System.err.print(vals[i] + " ");
			}
			System.err.println();
		}
	}

	public static void printDoubleArray(String tag, double[] vals, PrintStream out){
		out.print(tag + "[len:" + vals.length + "] ");
		for(int i = 0; i < vals.length; i++){
			out.print(vals[i] + " ");
		}
		out.println();
	}

	public static void printDoubleArray(String tag, double[] vals){
		printDoubleArray(tag, vals, true);
	}

	public static void printDoubleList(String tag, List<Double> vals, boolean debug) {		
		if(!disablePrints || !debug){
			System.err.print(tag + "[len:" + vals.size() + "] ");
			for(double val : vals){
				System.err.print(val + " ");
			}
			System.err.println();
		}
	}

	public static void printDoubleList(String tag, List<Double> vals) {		
		printDoubleList(tag, vals, true);
	}


	public static void printDoubleMatrix(String tag, double[][] vals){
		printDoubleMatrix(tag, vals, true);
	}

	public static void printDoubleMatrix(String tag, double[][] vals, boolean debug){
		if(!disablePrints || !debug){
			for(int i = 0; i < vals.length; i++){
				System.err.print(tag + " [" + i + "] ");
				for(int j = 0; j < vals[i].length; j++){
					System.err.print(vals[i][j] + " ");
				}
				System.err.println();
			}
			System.err.println();
		}
	}

	public static void printDoubleMatrix(String tag, double[][] vals, PrintStream out){
		for(int i = 0; i < vals.length; i++){
			out.print(tag + " [" + i + "] ");
			for(int j = 0; j < vals[i].length; j++){
				out.print(vals[i][j] + " ");
			}
			out.println();
		}
		out.println();

	}

	public static void printDoubleMatrix(String tag, int[][] vals){
		if(!disablePrints){
			for(int i = 0; i < vals.length; i++){
				System.err.print(tag + " [" + i + "] ");
				for(int j = 0; j < vals[i].length; j++){
					System.err.print(vals[i][j] + " ");
				}
				System.err.println();
			}
			System.err.println();
		}
	}



}
