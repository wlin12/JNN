package jnn.objective;

import jnn.neuron.DenseNeuronArray;
import util.ExpTable;

public class WordSoftmaxDenseObjective {

	int expectedIndex;	
	DenseNeuronArray neurons;
	double loglikelihood = -Double.MAX_VALUE;

	public WordSoftmaxDenseObjective(DenseNeuronArray neurons, int expectedIndex) {
		super();
		this.neurons = neurons;
		this.expectedIndex = expectedIndex;
	}

	public void addError(double norm) {
		for(int key = 0; key < neurons.size; key++){
			double label = 0;
			if(key == expectedIndex){
				label = 1;
			}
			neurons.addError(key, (label - ExpTable.getExpSing((neurons.getNeuron(key))))*norm);			
		}
	}
	
	public double getError(){
		double error = 0;
		for(int key = 0; key < neurons.size; key++){
			error += Math.abs(neurons.getError(key));			
		}
		return error/neurons.size;
	}
	
	public void addSoftmaxError(double norm){
		double max = neurons.getNeuron(expectedIndex);
		double[] expI = new double[neurons.size];
		int i = 0;
		double sumD = 0;
		double sum = 0;
		for(int key = 0; key < neurons.size; key++){
			double val = Math.exp(neurons.getNeuron(key) - max);
			expI[i++] = val;
			sumD+=val;
			sum+=Math.exp(neurons.getNeuron(key));
		}
		
		i = 0;
		for(int key = 0; key < neurons.size; key++){
			neurons.addError(key,(norm*(-expI[i++]/sumD)));
		}
		loglikelihood = neurons.getNeuron(expectedIndex) - Math.log(sum);
		neurons.addError(expectedIndex, norm);			
	}

	public double getLL() {
		return loglikelihood;
	}
}
