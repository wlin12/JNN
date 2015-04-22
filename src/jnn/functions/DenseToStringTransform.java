package jnn.functions;

import jnn.mapping.OutputMappingDenseToString;
import jnn.mapping.OutputMappingStringToDense;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.StringNeuronArray;

public interface DenseToStringTransform {
	public void forward(DenseNeuronArray input, int inputStart, int inputEnd, StringNeuronArray output, OutputMappingDenseToString mapping);
	public void backward(DenseNeuronArray input, int inputStart, int inputEnd, StringNeuronArray output, OutputMappingDenseToString mapping);
}
