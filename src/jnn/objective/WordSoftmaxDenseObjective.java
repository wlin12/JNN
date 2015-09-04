package jnn.objective;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.activation.Exp;
import org.nd4j.linalg.api.activation.SoftMax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import jnn.neuron.DenseNeuronArray;
import util.ExpTable;
import util.ExpTable.OutOfBoundariesException;
import util.LogAdd;

public class WordSoftmaxDenseObjective {

	int expectedIndex;	
	DenseNeuronArray neurons;
	double loglikelihood = -Double.MAX_VALUE;
	public static SoftMax exp = new SoftMax();

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
		INDArray expArray = ExpTable.getExpNormTable(neurons.getOutputRange(0, neurons.size-1));
		loglikelihood = FastMath.log(expArray.getDouble(expectedIndex));
		expArray.muli(-norm);		
		neurons.setErrorRange(0, neurons.size-1, expArray);
		neurons.addError(expectedIndex, norm);			
	}

	public void addNegativeSoftmaxError(double norm){
		//		INDArray expArray = exp.apply(neurons.getOutputRange(0, neurons.size-1));
		//		loglikelihood = FastMath.log(expArray.getDouble(expectedIndex));
		//		neurons.addError(expectedIndex, -norm);

		INDArray expArray = exp.apply(neurons.getOutputRange(0, neurons.size-1));
		double wordProb = 1/(double)(neurons.len());
		INDArray errorArray = expArray.mul(-norm).add(wordProb);	
		neurons.setErrorRange(0, neurons.size-1, errorArray);
		loglikelihood = FastMath.log(expArray.getDouble(expectedIndex));
	}

	public static double[] getLikelihoodArray(DenseNeuronArray neurons){
		double sum = 0;
		for(int key = 0; key < neurons.size; key++){
			sum += FastMath.exp(neurons.getNeuron(key));
		}
		double logsum = FastMath.log(sum);
		double[] ret = new double[neurons.size];
		for(int key = 0; key < neurons.size; key++){
			ret[key] = neurons.getNeuron(key) - logsum;
		}
		return ret;
	}

	public double getLL() {
		return loglikelihood;
	}

	public static void main(String[] args){
		DenseNeuronArray input = new DenseNeuronArray(10);
		input.init();
		for(int i = 0; i < 10; i++){
			input.addNeuron(i, i);
		}
		WordSoftmaxDenseObjective obj = new WordSoftmaxDenseObjective(input, 2);
		obj.addSoftmaxError(1);
		System.err.println(input);
		System.err.println(obj.getLL());

		INDArray inputArray = input.getOutputRange(0, 9);
		SoftMax sm = new SoftMax();
		INDArray outputArray = sm.apply(inputArray);
		INDArray expectedArray = Nd4j.zeros(10);
		expectedArray.putScalar(2, 1);
		INDArray errorArray = expectedArray.sub(outputArray);
		System.err.println(outputArray);
		System.err.println(expectedArray.sub(sm.apply(inputArray)));

	}
}
