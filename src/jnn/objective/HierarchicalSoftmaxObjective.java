package jnn.objective;

import java.util.HashMap;
import java.util.Map.Entry;

import jnn.neuron.SparseNeuronArray;

public class HierarchicalSoftmaxObjective implements Objective{
	public SparseNeuronArray neurons;	
	HashMap<Integer, Double> expected;
	
	public HierarchicalSoftmaxObjective(SparseNeuronArray neurons) {
		this.neurons = neurons;
	}
	
	@Override
	public void addError() {
		for(Entry<Integer, Double> entry : expected.entrySet()){
			neurons.addError(entry.getKey(), entry.getValue() - neurons.getOutput(entry.getKey()));			
		}			
	}
	
	public void setExpected(HashMap<Integer, Double> expected) {
		this.expected = expected;
	}

	public double getSquareError() {
		double error = 0;		
		for(Entry<Integer, Double> entry : expected.entrySet()){
			error+= (entry.getValue() - neurons.getOutput(entry.getKey()))*(entry.getValue() - neurons.getOutput(entry.getKey()));
		}
		return error/expected.size();
	}
}
