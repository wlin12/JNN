package jnn.functions;

import jnn.mapping.OutputMappingDenseArrayToStringArray;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.StringNeuronArray;

public interface DenseArrayToStringArrayTransform {
	public void forward(DenseNeuronArray[] input, int inputStart, int inputEnd, StringNeuronArray[] output, OutputMappingDenseArrayToStringArray mapping);
	public void backward(DenseNeuronArray[] input, int inputStart, int inputEnd, StringNeuronArray[] output, OutputMappingDenseArrayToStringArray mapping);
}
