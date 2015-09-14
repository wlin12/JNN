package jnn.objective;

import java.util.HashMap;
import java.util.Map.Entry;

import org.apache.commons.math3.util.FastMath;

import jnn.neuron.DenseNeuronArray;
import jnn.neuron.SparseNeuronArray;
import util.ExpTable;
import util.LogAdd;
import util.MathUtils;
import util.TopNList;

public class WordSoftmaxSparseObjective {

	public static final double MAX_VAL = 100;

	int expectedIndex;	
	SparseNeuronArray neurons;
	double logLL;
	HashMap<Integer, Double> indexes;
	
	public WordSoftmaxSparseObjective(SparseNeuronArray neurons, int expectedIndex, HashMap<Integer, Double> keys) {
		super();
		this.neurons = neurons;
		this.expectedIndex = expectedIndex;
		this.indexes = keys;
	}

	public void addError(double norm) {		
		for(Entry<Integer, Double> nonZero : neurons.getNonZeroEntries()){
			int key = nonZero.getKey();
			double label = 0;
			if(key == expectedIndex){
				label = 1;
			}
			double activation = neurons.getOutput(key);
			if(activation > ExpTable.SingletonMaxExp){
				neurons.addError(key, (label - 1)*norm);
			}
			else if(activation < -ExpTable.SingletonMaxExp){
				neurons.addError(key, (label)*norm);
			}
			else{
				neurons.addError(key, (label - ExpTable.getLogistic(activation))*norm);
			}
		}
	}
	
	public void addErrorNCE(double norm, int k, double[] noiseEstimation){		
		for(Entry<Integer, Double> nonZero : neurons.getNonZeroEntries()){
			int key = nonZero.getKey();
			double activation = nonZero.getValue();
			double pReal = 0;
			if(activation > 50){
				pReal = 1;
				nonZero.setValue(50.0);
			}
			else if(activation < -50){
				pReal = 0;
				nonZero.setValue(-50.0);
			}
			else{ 
				double act_exp = FastMath.exp(activation);
				pReal = act_exp/(noiseEstimation[key]*k+act_exp);
			}
			
			double pNoise = 1-pReal;
			
			if(key == expectedIndex){
				//System.err.println("real"+pReal);
				logLL+=FastMath.log(pReal);
				neurons.addError(key, (1-pReal)*norm);
			}
			else{
				//System.err.println("noise"+pReal);
				logLL+=FastMath.log(pNoise);
				neurons.addError(key, -pReal*norm*indexes.get(key));
			}
			//System.err.println(activation);
			//System.err.println(noiseEstimation[key]);
//			if(Double.isInfinite(logLL)){
//				System.err.println("loglikelihood was infinite");
//				System.exit(0);
//			}
		}	
		logLL/=k+1;
	}
	
	public void computeNCELoglikelihood(int k, double[] noiseEstimation){
		double sum = 0;
		double expected = 0;
		for(Entry<Integer, Double> nonZero : neurons.getNonZeroEntries()){
			int key = nonZero.getKey();
			double activation = nonZero.getValue();
			activation = FastMath.exp(activation);
			double pReal = 0;
			if(activation > 50){
				pReal = 1;
			}
			else if(activation < -50){
				pReal = 0;
			}
			else{ 
				double act_exp = FastMath.exp(activation);
				pReal = act_exp/(noiseEstimation[key]*k+act_exp);
			}
			
			if(expectedIndex == key) {
				expected = pReal;			
				sum+=pReal;
			}
			else{
				sum+=pReal*indexes.get(key);				
			}
		}
		logLL = FastMath.log(expected/sum);
	}
	
	public void computeLoglikelihood(int k, double[] noiseEstimation){
		double logsum = FastMath.log(0);
		double expected = 0;
		for(int i = 0; i < noiseEstimation.length; i++){
			double activation = neurons.getOutput(i);
			if(expectedIndex == i) {
				expected = activation;
				//System.err.println(i+" true "+activation);
			}
			else{
				//System.err.println(i+" false "+activation);
			}
			logsum = LogAdd.logAdd(logsum, activation);
		}
		logLL = expected - logsum;
	}
	
	public int getNCEMaxIndex(int k, double[] noiseEstimation) {
		int maxIndex = -1;
		double max = -Double.MAX_VALUE;
		for(int i = 0; i < noiseEstimation.length; i++){
			double activation = neurons.getOutput(i);
			//activation = FastMath.exp(activation);
			
			//double pReal = activation/(noiseEstimation[i]*k+activation);
			if(activation > max){
				maxIndex = i;
				max = activation;
			}
		}
		return maxIndex;
	}
	
	public TopNList<Integer> getNCETopN(int n, int k, double[] noiseEstimation) {
		double logsum = FastMath.log(0);
		for(int i = 0; i < noiseEstimation.length; i++){
			double activation = neurons.getOutput(i);
			logsum = LogAdd.logAdd(logsum, activation);
		}
		
		TopNList<Integer> topN = new TopNList<Integer>(n);
		for(int i = 0; i < noiseEstimation.length; i++){			
			double activation = neurons.getOutput(i);
			topN.add(i, activation-logsum);
		}
		return topN;
	}

	
	public double getError(){
		double error = 0;
		for(Entry<Integer, Double> nonZero : neurons.getNonZeroEntries()){
			error += Math.abs(neurons.getError(nonZero.getKey()));
		}
		return error/neurons.getNonZeroEntries().size();
	}
	
	public void addSoftmaxError(double norm){
		double max = neurons.getOutput(neurons.maxIndex());
		double[] expI = new double[neurons.size];
		int i = 0;
		for(Entry<Integer, Double> nonZero : neurons.getNonZeroEntries()){
			expI[i++] = Math.exp(nonZero.getValue() - max);
		}
		
		i = 0;
		double sum = MathUtils.sum(expI);
		for(Entry<Integer, Double> nonZero : neurons.getNonZeroEntries()){
			neurons.addError(nonZero.getKey(),(norm*(-expI[i++]/sum)));
		}
		neurons.addError(expectedIndex, norm);			
	}
	
	public double getLL() {
		return logLL;
	}

	public int getMaxIndex() {
		double max = -Double.MAX_VALUE;
		int index = -1;
		for(Entry<Integer, Double> nonZero : neurons.getNonZeroEntries()){
			if(nonZero.getValue() > max){
				max = nonZero.getValue();
				index = nonZero.getKey();
			}
		}
		return index;
	}
	

	public TopNList<Integer> getTopN(int n) {
		TopNList<Integer> topN = new TopNList<Integer>(n);
		for(Entry<Integer, Double> nonZero : neurons.getNonZeroEntries()){
			topN.add(nonZero.getKey(), nonZero.getValue());
		}
		return topN;
	}
}
