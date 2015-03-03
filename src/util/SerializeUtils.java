package util;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Scanner;

public class SerializeUtils {
	public static void saveDoubleMatrix(double[][] matrix, PrintStream out){
		int rows = matrix.length;
		int cols = matrix[0].length;
		out.println(rows + " " + cols);
		for(int i = 0; i < rows; i++){
			for(int j = 0; j < cols; j++){
				out.print(matrix[i][j]);
				if(j != cols - 1){
					out.print(" ");
				}
			}
			out.println();
		}
	}
	
	public static void saveDoubleMatrixFromBinary(DataInputStream reader,
			int rows, int cols, PrintStream out) {
		out.println(rows + " " + cols);
		for(int i = 0; i < rows; i++){
			double[] row = readDoubleArrayFromFileBinary(reader, cols);
			for(int j = 0; j < cols; j++){
				out.print(row[j]);
				if(j != cols - 1){
					out.print(" ");
				}
			}
			out.println();
		}
	}

	public static double[][] loadDoubleMatrix(Scanner reader){
		int rows = reader.nextInt();
		int cols = reader.nextInt();
		double[][] ret = new double[rows][cols];
		for(int i = 0; i < rows; i++){
			for(int j = 0; j < cols; j++){
				ret[i][j] = Double.parseDouble(reader.next());
			}
		}
		return ret;
	}

	public static void saveDoubleArray(double[] array, PrintStream out){
		int dim = array.length;
		out.println(dim);
		for(int i = 0; i < dim; i++){
			out.print(array[i]);
			if(i != dim - 1){
				out.print(" ");
			}
		}
		out.println();
	}	
	
	public static void saveLongArray(long[] array, PrintStream out){
		int dim = array.length;
		out.println(dim);
		for(int i = 0; i < dim; i++){
			out.print(array[i]);
			if(i != dim - 1){
				out.print(" ");
			}
		}
		out.println();
	}	
	

	public static void saveIntArray(int[] array, PrintStream out) {
		int dim = array.length;
		out.println(dim);
		for(int i = 0; i < dim; i++){
			out.print(array[i]);
			if(i != dim - 1){
				out.print(" ");
			}
		}
		out.println();
	}
	
	public static void saveBooleanArray(boolean[] array, PrintStream out) {
		int dim = array.length;
		out.println(dim);
		for(int i = 0; i < dim; i++){
			out.print(array[i]);
			if(i != dim - 1){
				out.print(" ");
			}
		}
		out.println();
	}
	

	public static void saveStringDoubleMap(Map<String, double[]> map, int dim, PrintStream out) {
		out.println(map.size() + " " + dim);
		for(Entry<String, double[]> entry : map.entrySet()){
			out.println(entry.getKey());
			for(int i = 0; i < dim; i++){
				out.print(entry.getValue()[i]);
				if(i != dim - 1){
					out.print(" ");
				}
			}
			out.println();
		}
	}

	public static double[] loadDoubleArray(Scanner reader){
		int dim = reader.nextInt();
		double[] ret = new double[dim];
		for(int i = 0; i < dim; i++){			
			ret[i] = Double.parseDouble(reader.next());
		}
		return ret;
	}
	
	public static int[] loadIntArray(Scanner reader){
		int dim = reader.nextInt();
		int[] ret = new int[dim];
		for(int i = 0; i < dim; i++){			
			ret[i] = Integer.parseInt(reader.next());
		}
		return ret;
	}
	
	public static int[] loadIntArray(BufferedReader in){
		try {
			int dim = Integer.parseInt(in.readLine());
			String[] vals = in.readLine().split("\\s+");
			int[] ret = new int[dim];
			for(int i = 0; i < ret.length; i++){
				ret[i] = Integer.parseInt(vals[i]);
			}
			return ret;
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	public static boolean[] loadBooleanArray(BufferedReader in){
		try {
			int dim = Integer.parseInt(in.readLine());
			String[] vals = in.readLine().split("\\s+");
			boolean[] ret = new boolean[dim];
			for(int i = 0; i < ret.length; i++){
				ret[i] = Boolean.parseBoolean(vals[i]);
			}
			return ret;
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
	
	public static long[] loadLongArray(BufferedReader in){
		try {
			int dim = Integer.parseInt(in.readLine());
			String[] vals = in.readLine().split("\\s+");
			long[] ret = new long[dim];
			for(int i = 0; i < ret.length; i++){
				ret[i] = Long.parseLong(vals[i]);
			}
			return ret;
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
	
	public static boolean[] loadBooleanArray(Scanner reader){
		int dim = reader.nextInt();
		boolean[] ret = new boolean[dim];
		for(int i = 0; i < dim; i++){			
			ret[i] = Boolean.parseBoolean(reader.next());
		}
		return ret;
	}
	
	public static double[] loadDoubleArray(Scanner reader, int size){
		double[] ret = new double[size];
		for(int i = 0; i < size; i++){
			ret[i] = Double.parseDouble(reader.next());
		}
		return ret;
	}
	
	public static HashMap<String, double[]> loadStringDoubleMap(Scanner scanner) {
		HashMap<String, double[]> ret = new HashMap<String, double[]>();
		int dimX = scanner.nextInt();
		int dimY = scanner.nextInt();
		scanner.nextLine();
		for(int i = 0; i < dimX; i++){			
			String word = scanner.nextLine();
			double[] vals = new double[dimY];
			for(int j = 0; j < dimY; j++){
				vals[j] = scanner.nextDouble();
			}
			scanner.nextLine();
			ret.put(word, vals);
		}
		return ret;

	}

	public static void saveNDimDoubleMatrix(Object nmatrix, int[] dim, PrintStream out){
		for(int i = 0; i < dim.length; i++){
			out.print(dim[i]);
			if(i != dim.length-1){
				out.print(" ");
			}
		}
		out.println();
		saveNDimDoubleMatrixRec(nmatrix, dim, out);
	}

	private static void saveNDimDoubleMatrixRec(Object nmatrix, int[] dim, PrintStream out){
		if(dim.length == 1){
			int currentDim = dim[0];
			double[] array = (double[]) nmatrix;
			for(int i = 0; i < currentDim; i++){
				out.print(array[i]);
				if(i != currentDim - 1){
					out.print(" ");
				}
			}
			out.println();
		}
		else{
			int[] newDim = Arrays.copyOfRange(dim, 1, dim.length);
			for(int i = 0; i < dim[0]; i++){
				saveNDimDoubleMatrixRec(((Object[])nmatrix)[i], newDim, out);
			}
		}
	}
	
	public static int[] loadNDimDoubleMatrix(Scanner reader, int numberOfDim, Object nmatrix){
		int[] dim = new int[numberOfDim];
		for(int i = 0; i < dim.length; i++){
			dim[i] = Integer.parseInt(reader.next());
		}
		return dim;
	}

	public static void loadNDimDoubleMatrixRec(Scanner reader, int[] dim, Object nmatrix){
		if(dim.length == 1){
			for(int i = 0; i < dim[0];i++){
				((double[])nmatrix)[i] = reader.nextDouble();
			}
		}
		else{
			int[] newDim = Arrays.copyOfRange(dim, 1, dim.length);			
			for(int i = 0; i < dim[0]; i++){
				loadNDimDoubleMatrixRec(reader, newDim, ((Object[])nmatrix)[i]);
			}
		}
	}

	public static double[] readDoubleArrayFromFileBinary(
			DataInputStream storeReader, int dimensions) {
		double[] ret = new double[dimensions];
		for(int i = 0; i < dimensions; i++){
			try {
				ret[i] = storeReader.readDouble();
			} catch (EOFException e){
				return null;
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}
		return ret;
	}

	public static void writeDoubleArrayToFileBinary(DataOutputStream storeWriter, double[] array) {
		for(int i = 0; i < array.length; i++){
			try {
				storeWriter.writeDouble(array[i]);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}



}
