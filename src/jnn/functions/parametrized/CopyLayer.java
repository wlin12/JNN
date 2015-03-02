package jnn.functions.parametrized;

import java.util.Set;

import jnn.functions.DenseToDenseTransform;
import jnn.functions.SparseToSparseTransform;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.mapping.OutputMappingSparseToSparse;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.SparseNeuronArray;

public class CopyLayer extends Layer implements DenseToDenseTransform, SparseToSparseTransform{
	
	public static CopyLayer singleton = new CopyLayer();
	
	@Override
	public void forward(DenseNeuronArray input, int inputStart, int inputEnd,
			DenseNeuronArray output, int outputStart, int outputEnd, OutputMappingDenseToDense mapping) {
		int inputDim = inputEnd - inputStart + 1;
		int outputDim = outputEnd - outputStart + 1;
		if(inputDim != outputDim){
			throw new RuntimeException(inputDim + " " + outputDim);
		}
		for(int i = 0; i < inputDim; i++){
			output.addNeuron(i+outputStart, input.getNeuron(i+inputStart));
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
			output.addNeuron(i+outputStart,input.getOutput(i+inputStart));
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
		for(int i = 0; i < inputDim; i++){	
			input.addError(i+inputStart, output.getError(i+outputStart));
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
			input.addError(i+inputStart, output.getError(i+outputStart));
		}
	}
}
