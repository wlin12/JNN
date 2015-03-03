package jnn.objective;

import java.util.Map.Entry;

import jnn.neuron.SparseNeuronArray;
import util.ExpTable;
import util.MathUtils;

public class WordSoftmaxSparseObjective {

	public static final double MAX_VAL = 100;

	int expectedIndex;	
	SparseNeuronArray neurons;
	
	public WordSoftmaxSparseObjective(SparseNeuronArray neurons, int expectedIndex) {
		super();
		this.neurons = neurons;
		this.expectedIndex = expectedIndex;
	}

	public void addError(double norm) {		
		for(Entry<Integer, Double> nonZero : neurons.getNonZeroEntries()){
			int key = nonZero.getKey();
			double label = 0;
			if(key == expectedIndex){
				label = 1;
			}
			double activation = neurons.getOutput(key);
			if(activation > MAX_VAL){
				activation = MAX_VAL;
			}
			else if(activation < -MAX_VAL){
				activation = -MAX_VAL;
			}
			neurons.addError(key, (label - ExpTable.getExpSing(activation))*norm);
		}
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
}
