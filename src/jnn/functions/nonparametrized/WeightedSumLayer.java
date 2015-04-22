package jnn.functions.nonparametrized;

import java.util.Set;

import jnn.functions.DenseArrayToDenseTransform;
import jnn.functions.DenseToDenseTransform;
import jnn.functions.SparseToSparseTransform;
import jnn.functions.parametrized.Layer;
import jnn.mapping.OutputMappingDenseArrayToDense;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.mapping.OutputMappingSparseToSparse;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.SparseNeuronArray;
import jnn.training.GraphInference;

import org.nd4j.linalg.api.activation.ActivationFunction;
import org.nd4j.linalg.api.activation.Sigmoid;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import util.ExpTable;

public class WeightedSumLayer extends Layer implements DenseArrayToDenseTransform{
	
	public static WeightedSumLayer singleton = new WeightedSumLayer();

	@Override
	public void forward(DenseNeuronArray[] input, int inputStart, int inputEnd,
			DenseNeuronArray output, int outputStart, int outputEnd,
			OutputMappingDenseArrayToDense mapping) {
		
		int inputDim = inputEnd - inputStart + 1;
		int outputDim = outputEnd - outputStart + 1;
		if(inputDim != outputDim){
			throw new RuntimeException(inputDim + " " + outputDim);
		}
		
		DenseNeuronArray weights = input[input.length-1];
		if(weights.size != input.length-1){
			throw new RuntimeException("Incorrect input for this layer: there are " + weights.size + " weights and " + (input.length-1) + " inputs");
		}		
		
		INDArray sum = input[0].getOutputRange(inputStart, inputEnd).mul(weights.getNeuron(0));
		for(int i = 1; i < input.length-1; i++){
			sum.addi(input[i].getOutputRange(inputStart, inputEnd).mul(weights.getNeuron(i)));
		}
		
		output.setOutputRange(outputStart, outputEnd, sum);
	}
	
	@Override
	public void backward(DenseNeuronArray[] input, int inputStart,
			int inputEnd, DenseNeuronArray output, int outputStart,
			int outputEnd, OutputMappingDenseArrayToDense mapping) {
		DenseNeuronArray weights = input[input.length-1];
		
		INDArray sumGrad = output.getErrorRange(outputStart, outputEnd);
		for(int i = 0; i < input.length-1; i++){
			input[i].setErrorRange(inputStart, inputEnd, sumGrad.mul(weights.getNeuron(i)));
			weights.addError(i, input[i].getOutputRange(inputStart, inputEnd).mul(sumGrad).sum(0).getDouble(0));
		}
	}	

	public static void main(String[] args){
		DenseNeuronArray inputArray = new DenseNeuronArray(5);
		inputArray.init();
		inputArray.addNeuron(0, 0.1);
		inputArray.addNeuron(1, 0.2);
		inputArray.addNeuron(2, 0.3);
		inputArray.addNeuron(3, 0.2);
		inputArray.addNeuron(4, 0.3);
		DenseNeuronArray inputArray2 = new DenseNeuronArray(5);
		inputArray2.init();
		inputArray2.addNeuron(0, -0.1);
		inputArray2.addNeuron(1, -0.3);
		inputArray2.addNeuron(2, 0.2);
		inputArray2.addNeuron(3, 0.1);
		inputArray2.addNeuron(4, 0.2);
		
		DenseNeuronArray weights = new DenseNeuronArray(2);
		weights.init();
		weights.addNeuron(0, 0.3);
		weights.addNeuron(1, 0.7);
		
		DenseNeuronArray weightedArray = new DenseNeuronArray(5);		
		
		GraphInference inference = new GraphInference(0, true);
		inference.addNeurons(0,inputArray);
		inference.addNeurons(0,inputArray2);
		inference.addNeurons(0,weights);
		inference.addNeurons(1,weightedArray);
		
		
		inference.addMapping(new OutputMappingDenseArrayToDense(new DenseNeuronArray[]{inputArray, inputArray2, weights}, weightedArray, WeightedSumLayer.singleton));;
		inference.init();
		inference.forward();
				
		weightedArray.addError(0, -0.1);
		weightedArray.addError(1, 0.2);
		
		inference.backward();
		inference.printNeurons();
	}
}
