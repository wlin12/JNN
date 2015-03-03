package util;

import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map.Entry;
import java.util.Set;

import org.nd4j.linalg.api.ndarray.INDArray;

public class MathUtils {

	public static double[] normVector(double[] vector) {
		double[] norm = new double[vector.length];
		double len = 0;
		for(int j = 0; j < vector.length; j++){
			len += vector[j] * vector[j]; 
		}
		len = Math.sqrt(len);
		for(int j = 0; j < vector.length; j++){
			norm[j] = vector[j]/len;
		}
		return norm;
	}
	
	public static double[] normVectorTo1(double[] vector) {
		double[] norm = new double[vector.length];
		double len = 0;
		for(int j = 0; j < vector.length; j++){
			len += vector[j]; 
		}
		for(int j = 0; j < vector.length; j++){
			norm[j] = vector[j]/len;
		}
		return norm;
	}

	public static int maxIndex(double[] vector) {
		int maxIndex = 0;
		for(int i = 1; i < vector.length; i++){
			if (vector[i] > vector[maxIndex]){
				maxIndex = i;
			}
		}
		return maxIndex;
	}
	
	public static int maxIndex(int[] vector) {
		int maxIndex = 0;
		for(int i = 1; i < vector.length; i++){
			if (vector[i] > vector[maxIndex]){
				maxIndex = i;
			}
		}
		return maxIndex;
	}
	
	public static int minIndex(int[] vector) {
		int maxIndex = 0;
		for(int i = 1; i < vector.length; i++){
			if (vector[i] < vector[maxIndex]){
				maxIndex = i;
			}
		}
		return maxIndex;
	}
	
	public static int maxIndex(double[] vector, int start, int end) {
		int maxIndex = start;
		for(int i = start+1; i <= end; i++){
			if (vector[i] > vector[maxIndex]){
				maxIndex = i;
			}
		}
		return maxIndex;
	}

	
	public static double max(double[] vector){
		return vector[maxIndex(vector)];
	}

	public static double arraySum(double[] vector) {
		double ret = 0;
		for(double d : vector){
			ret+=d;
		}
		return ret;
	}
	
	public static int sum(int[] vector){
		int ret = 0;
		for(int d : vector){
			ret+=d;
		}
		return ret;		
	}
	
	public static double arraySumAbs(double[] vector) {
		double ret = 0;
		for(double d : vector){
			ret+=Math.abs(d);
		}
		return ret;
	}

	public static void copyArray(double[] from, int fromStart, int fromEnd,
			double[] to, int offset) {
		for(int i = 0; i <= fromEnd - fromStart; i++){
			to[i + offset] = from[i + fromStart]; 
		}
	}

	public static double[] addVectors(double[] a, double[] b) {
		double[] ret = new double[a.length];
		for(int i = 0; i < a.length; i++){
			ret[i] = a[i] + b[i];
		}
		return ret;
	}
	
	public static double[] subVectors(double[] a, double[] b) {
		double[] ret = new double[a.length];
		for(int i = 0; i < a.length; i++){
			ret[i] = a[i] - b[i];
		}
		return ret;
	}
	
	public static void addVectors(double[] a, double[] b, double[] out) {
		if(a==null){
			throw new RuntimeException("a is null");
		}
		if(b==null){
			throw new RuntimeException("b is null");
		}
		if(out==null){
			throw new RuntimeException("out is null");
		}
		for(int i = 0; i < a.length; i++){
			out[i] = a[i] + b[i];
		}
	}
	
	public static void subVectors(double[] a, double[] b, double[] out) {
		for(int i = 0; i < a.length; i++){
			out[i] = a[i] - b[i];
		}
	}

	public static int drawFromCategorical(double[] categoricalDist) {
		double[] normalized = normVectorTo1(categoricalDist);
		double rand = Math.random();
		double sum = 0;
		for(int i = 0; i < normalized.length - 1; i++){
			sum += normalized[i];
			if(rand < sum){
				return i;
			}
		}
		return normalized.length - 1;
	}
	
	public static String drawFromCategorical(HashMap<String, Double> categoricalDist) {
		double rand = Math.random();
		double sum = 0;
		String last = "";
		for(Entry<String, Double> entry : categoricalDist.entrySet()){
			sum += entry.getValue();
			if(rand < sum){
				return entry.getKey();
			}
			last = entry.getKey();
		}
		return last;
	}

	public static String drawFromUniform(Set<String> outputs) {
		double rand = Math.random();
		double sum = 0;
		double val = 1.0/outputs.size();
		String last = "";
		for(String key : outputs){
			sum += val;
			if(rand < sum){
				return key;
			}
			last = key;
		}
		return last;
	}

	public static double sum(double[] ds) {
		double sum = 0;
		for(double d:ds){
			sum+=d;
		}
		return sum;
	}
	
	public static double sumSquared(double[] ds) {
		double sum = 0;
		for(double d:ds){
			sum+=d*d;
		}
		return Math.sqrt(sum);
	}

	public static void zero(double[] output, int start, int end) {
		for(int i = start; i <= end;i++ ){
			output[i] = 0;
		}
	}

	public static double[] scale(double[] subVectors, double alpha) {
		double[] ret = new double[subVectors.length];
		for(int i = 0; i < ret.length; i++){
			ret[i] = subVectors[i]*alpha;
		}
		return ret;
	}

	public static double[] concatVectors(LinkedList<double[]> vectors) {
		int size = 0;
		for(double[] vector : vectors){
			size+=vector.length;
		}
		double[] ret = new double[size];
		concatVectors(vectors, ret);
		return ret;
	}

	public static double[] concatVectors(LinkedList<double[]> vectors, double[] ret) {
		int i = 0;
		for(double[] vector : vectors){
			for(double d : vector){
				ret[i++] = d;
			}
		}
		return ret;
	}


	public static double[] collapse(double[][] map) {
		double[] ret = new double[map.length * map[0].length];
		for(int i = 0; i < map.length; i++){
			for(int j = 0; j < map[0].length; j++){
				ret[i*map[0].length + j] = map[i][j]; 
			}
		}
		return ret;
	}
	

	public static double[] collapse(LinkedList<double[]> previousInput) {
		double[] ret = new double[previousInput.size() * previousInput.getFirst().length];
		for(int i = 0; i < previousInput.size(); i++){
			for(int j = 0; j < previousInput.getFirst().length; j++){
				ret[i*previousInput.getFirst().length + j] = previousInput.get(i)[j]; 
			}
		}
		return ret;
	}


	public static double[][] arrayToMatrix(double[] vector, int col){
		if(vector.length % col != 0){
			throw new RuntimeException("incorrect vector size");
		}
		int dimY = vector.length / col; 
		double[][] matrix = new double[col][dimY];
		for(int i = 0; i < col; i++){
			for(int j = 0; j < dimY; j++){
				matrix[i][j] = vector[i*dimY+j];
			}
		}
		return matrix;
	}

	public static double getLearningRate(double initAlpha, double finalAlpha, double currentIteration, double currentSample, double lastIteration, double lastSample){
		double progression = (currentIteration*lastSample+currentSample)/(lastIteration*lastSample);
		return initAlpha - (initAlpha - finalAlpha)*progression;
	}
	
	public static void checkArraySize(double[] input, int dim){
		if(input.length != dim){
			throw new RuntimeException("incorrect input dim size: input = " + input.length + " expected = " + dim);
		}
	}

	public static int minIndex(Iterator<Entry<Integer, Double>> iterator) {
		double minVal = Double.MAX_VALUE;
		int minIndex = -1;
		while(iterator.hasNext()){
			Entry<Integer, Double> next = iterator.next();
			if(next.getValue()<minVal){
				minIndex = next.getKey();
				minVal = next.getValue();
			}
		}
		return minIndex;
	}

	public static int maxIndex(INDArray outputs) {
		double minVal = -Double.MAX_VALUE;
		int minIndex = -1;
		for(int i = 0; i < outputs.columns(); i++){
			double val = outputs.getDouble(i);
			if(minVal<val){
				minIndex = i;
				minVal = val;
			}
		}
		return minIndex;
	}

	public static double max(INDArray outputs) {
		return outputs.getDouble(maxIndex(outputs));
	}
	
	public static void cap(INDArray outputs, double max){
		INDArray linear = outputs.linearView();
		for(int i = 0; i < linear.size(0); i++){
			double val = linear.getDouble(i);
			if(val > max){
				linear.putScalar(i, max);
				continue;
			}
			if(val < -max){
				linear.putScalar(i, -max);				
			}
		}
	}
}
