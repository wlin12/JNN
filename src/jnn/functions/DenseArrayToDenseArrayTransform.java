package jnn.functions;

import jnn.mapping.OutputMappingDenseArrayToDenseArray;
import jnn.neuron.DenseNeuronArray;

public interface DenseArrayToDenseArrayTransform {
	public void forward(DenseNeuronArray[] input, int inputStart, int inputEnd, DenseNeuronArray[] output, int outputStart, int outputEnd, OutputMappingDenseArrayToDenseArray mapping);
	public void backward(DenseNeuronArray[] input, int inputStart, int inputEnd, DenseNeuronArray[] output, int outputStart, int outputEnd,OutputMappingDenseArrayToDenseArray mapping);
}
