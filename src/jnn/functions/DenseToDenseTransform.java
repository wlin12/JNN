package jnn.functions;

import jnn.mapping.OutputMappingDenseToDense;
import jnn.neuron.DenseNeuronArray;

public interface DenseToDenseTransform {	
	public void forward(DenseNeuronArray input, int inputStart, int inputEnd, DenseNeuronArray output, int outputStart, int outputEnd, OutputMappingDenseToDense mapping);
	public void backward(DenseNeuronArray input, int inputStart, int inputEnd, DenseNeuronArray output, int outputStart, int outputEnd, OutputMappingDenseToDense mapping);
}
