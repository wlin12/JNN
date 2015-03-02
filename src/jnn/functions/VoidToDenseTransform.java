package jnn.functions;

import jnn.mapping.OutputMappingVoidToDense;
import jnn.neuron.DenseNeuronArray;

public interface VoidToDenseTransform {
	public void forward(DenseNeuronArray output, int outputStart, int outputEnd, OutputMappingVoidToDense mapping);
	public void backward(DenseNeuronArray output, int outputStart, int outputEnd, OutputMappingVoidToDense mapping);
}
