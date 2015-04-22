package jnn.functions;

import jnn.mapping.OutputMappingStringToDense;
import jnn.mapping.OutputMappingStringArrayToDenseArray;
import jnn.neuron.DenseNeuronArray;

public interface StringArrayToDenseArrayTransform {
	public void forward(String[] input, DenseNeuronArray[] output, int outputStart, int outputEnd, OutputMappingStringArrayToDenseArray mapping);
	public void backward(String[] input, DenseNeuronArray[] output, int outputStart, int outputEnd, OutputMappingStringArrayToDenseArray mapping);

}
