package jnn.functions.nonparametrized;

import java.util.Set;

import jnn.functions.DenseToDenseTransform;
import jnn.functions.SparseToSparseTransform;
import jnn.functions.parametrized.Layer;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.mapping.OutputMappingSparseToSparse;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.SparseNeuronArray;

import org.nd4j.linalg.api.activation.ActivationFunction;
import org.nd4j.linalg.api.activation.HardTanh;

import util.TanFuncs;

public class TanHardSigmoidLayer extends Layer implements DenseToDenseTransform, SparseToSparseTransform{
	
	public static TanHardSigmoidLayer singleton = new TanHardSigmoidLayer();
	public static ActivationFunction activation = new HardTanh();

	@Override
	public void forward(DenseNeuronArray input, int inputStart, int inputEnd,
			DenseNeuronArray output, int outputStart, int outputEnd, OutputMappingDenseToDense mapping) {
		int inputDim = inputEnd - inputStart + 1;
		int outputDim = outputEnd - outputStart + 1;
		if(inputDim != outputDim){
			throw new RuntimeException(inputDim + " " + outputDim);
		}
//		output.setOutputRange(outputStart, outputEnd, activation.apply(input.getOutputRange(inputStart, inputEnd)));

		for(int i = 0; i < inputDim; i++){
			output.addNeuron(i+outputStart,TanFuncs.sigmoidHard(input.getNeuron(i+inputStart)));
		}
		
	}
	
	@Override
	public void forward(SparseNeuronArray input, int inputStart, int inputEnd,
			SparseNeuronArray output, int outputStart, int outputEnd, OutputMappingSparseToSparse mapping) {
		int inputDim = inputEnd - inputStart + 1;
		int outputDim = outputEnd - outputStart + 1;
		if(inputDim != outputDim){
			throw new RuntimeException(inputDim + " " + outputDim);
		}
		Set<Integer> indexes = input.getNonZeroKeys();
		for(int i : indexes){
			output.addNeuron(i+outputStart,TanFuncs.sigmoidHard(input.getOutput(i+inputStart)));
		}
		
	}
	
	@Override
	public void backward(DenseNeuronArray input, int inputStart, int inputEnd,
			DenseNeuronArray output, int outputStart, int outputEnd, OutputMappingDenseToDense mapping) {
		int inputDim = inputEnd - inputStart + 1;
		int outputDim = outputEnd - outputStart + 1;
		if(inputDim != outputDim){
			throw new RuntimeException(inputDim + " " + outputDim);
		}
//		INDArray x = input.getOutputRange(inputStart, inputEnd);		
//		INDArray yGrad = output.getErrorRange(outputStart, outputEnd);
//		INDArray xGrad = activation.applyDerivative(x).mul(yGrad);
//		input.setErrorRange(inputStart, inputEnd, xGrad);
		for(int i = 0; i < inputDim; i++){	
//			input.addError(i+inputStart, TanFuncs.dsigmoid(output.getNeuron(i+outputStart))*output.getError(i+outputStart));
			input.addError(i+inputStart, TanFuncs.dsigmoidHard(input.getNeuron(i+inputStart), output.getError(i+outputStart))*output.getError(i+outputStart));
		}
	}

	@Override
	public void backward(SparseNeuronArray input, int inputStart, int inputEnd,
			SparseNeuronArray output, int outputStart, int outputEnd, OutputMappingSparseToSparse mapping) {
		int inputDim = inputEnd - inputStart + 1;
		int outputDim = outputEnd - outputStart + 1;
		if(inputDim != outputDim){
			throw new RuntimeException(inputDim + " " + outputDim);
		}
		Set<Integer> indexes = input.getNonZeroKeys();
		for(int i : indexes){
//			input.addError(i+inputStart, TanFuncs.dsigmoid(output.getOutput(i+outputStart))*output.getError(i+outputStart));
			input.addError(i+inputStart, TanFuncs.dsigmoidHard(input.getOutput(i+inputStart), output.getError(i+outputStart))*output.getError(i+outputStart));
		}
	}
}
