package jnn.objective;

import jnn.neuron.DenseNeuronArray;
import util.MathUtils;

public class SoftmaxDenseObjective implements Objective{
	public DenseNeuronArray neurons;
	int expectedIndex = -1;
	int min;
	int max;
	double norm = 1;
	
	public SoftmaxDenseObjective(int max, int min, DenseNeuronArray neurons) {
		this.neurons = neurons;
		this.min = min;
		this.max = max;
	}
	
	public int getMaxIndex(){
		return neurons.maxIndex();
	}
	
	public void setExpectedIndex(int expectedIndex) {
		this.expectedIndex = expectedIndex;
	}
	
	@Override
	public void addError() {
		if(expectedIndex == -1){
			throw new RuntimeException("no expected index set");
		}
		for(int i = 0; i < neurons.len(); i++){
			double expected = min;			
			if(expectedIndex == i){
				expected = max;
			}
			neurons.addError(i, (expected - neurons.getNeuron(i))/((max-min)*norm));
		}		
	}

	public double getClassError() {
		if( expectedIndex != getMaxIndex()){
			return 1;
		}
		return 0;
	}
	
	public int getExpectedIndex() {
		return expectedIndex;
	}
	
	public double[] getProbs() {
		double[] probs = new double[neurons.size];
		
		for(int i = 0; i < probs.length; i++){
			probs[i] = (neurons.getNeuron(i) - min) / (max-min);
			if(probs[i]<0){
				throw new RuntimeException("invalid value i = " + i + " min = " + min + " max = " + max + " neuron activation = " + neurons.getNeuron(i));
			}
		}
		return MathUtils.normVectorTo1(probs);
	}

	public void printError() {
		if(expectedIndex != getMaxIndex()){
			System.err.println("error expected was " + expectedIndex + " got " + getMaxIndex());
		}
		else{
			System.err.println("correct gotten " + expectedIndex);
		}
	}
	
	public void setNorm(double norm) {
		this.norm = norm;
	}

	public double getSquareError() {
		double error = 0;
		for(int i = 0 ; i < neurons.size; i++){
			if(i == expectedIndex){
				error += (max - neurons.getNeuron(i)) * (max - neurons.getNeuron(i)); 
			}
			else{
				error += (min - neurons.getNeuron(i)) * (min - neurons.getNeuron(i)); 				
			}
		}
		return error/neurons.size;
	}
}
