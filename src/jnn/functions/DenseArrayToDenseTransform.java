package jnn.functions;

import jnn.mapping.OutputMappingDenseArrayToDense;
import jnn.neuron.DenseNeuronArray;

public interface DenseArrayToDenseTransform {	
	public void forward(DenseNeuronArray[] input, int inputStart, int inputEnd, DenseNeuronArray output, int outputStart, int outputEnd, OutputMappingDenseArrayToDense mapping);
	public void backward(DenseNeuronArray[] input, int inputStart, int inputEnd, DenseNeuronArray output, int outputStart, int outputEnd, OutputMappingDenseArrayToDense mapping);
		
}
