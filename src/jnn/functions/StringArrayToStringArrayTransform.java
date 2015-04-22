package jnn.functions;

import jnn.mapping.OutputMappingDenseToString;
import jnn.mapping.OutputMappingStringArrayToStringArray;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.StringNeuronArray;

public interface StringArrayToStringArrayTransform {
	public void forward(String[] input, StringNeuronArray[] output, OutputMappingStringArrayToStringArray mapping);
	public void backward(String[] input, StringNeuronArray[] output, OutputMappingStringArrayToStringArray mapping);

}
