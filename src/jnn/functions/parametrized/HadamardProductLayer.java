package jnn.functions.parametrized;

import jnn.functions.DenseArrayToDenseTransform;
import jnn.mapping.OutputMappingDenseArrayToDense;
import jnn.neuron.DenseNeuronArray;

import org.nd4j.linalg.api.ndarray.INDArray;

public class HadamardProductLayer extends Layer implements DenseArrayToDenseTransform{

	public static HadamardProductLayer singleton = new HadamardProductLayer();
	
	@Override
	public void forward(DenseNeuronArray[] input, int inputStart, int inputEnd,
			DenseNeuronArray output, int outputStart, int outputEnd,
			OutputMappingDenseArrayToDense mapping) {
		if(input.length != 2){
			throw new RuntimeException("incorrect input size. expected = 2");			
		}
		INDArray x1 = input[0].getOutputRange(inputStart, inputEnd);
		INDArray x2 = input[1].getOutputRange(inputStart, inputEnd);
		INDArray y = x1.mul(x2);
		output.setOutputRange(outputStart, outputEnd, y);		
	}

	@Override
	public void backward(DenseNeuronArray[] input, int inputStart,
			int inputEnd, DenseNeuronArray output, int outputStart,
			int outputEnd, OutputMappingDenseArrayToDense mapping) {
		INDArray x1 = input[0].getOutputRange(inputStart, inputEnd);
		INDArray x2 = input[1].getOutputRange(inputStart, inputEnd);
		INDArray yGrad = output.getErrorRange(outputStart, outputEnd);
		INDArray x1Grad = x2.mul(yGrad);
		INDArray x2Grad = x1.mul(yGrad);
		input[0].setErrorRange(inputStart, inputEnd, x1Grad);
		input[1].setErrorRange(inputStart, inputEnd, x2Grad);		
	}
	
	
}
