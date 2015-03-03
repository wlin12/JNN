package jnn.functions;

import jnn.mapping.OutputMappingStringToDense;
import jnn.neuron.DenseNeuronArray;

public interface StringToDenseTransform {
	public void forward(String input, DenseNeuronArray output, int outputStart, int outputEnd, OutputMappingStringToDense mapping);
	public void backward(String input, DenseNeuronArray output, int outputStart, int outputEnd, OutputMappingStringToDense mapping);

}
