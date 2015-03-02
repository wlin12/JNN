package jnn.functions;

import jnn.mapping.OutputMappingSparseArrayToDense;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.SparseNeuronArray;

public interface SparseArrayToDenseTransform {
	public void forward(SparseNeuronArray[] input, int inputStart, int inputEnd, DenseNeuronArray output, int outputStart, int outputEnd, OutputMappingSparseArrayToDense mapping);
	public void backward(SparseNeuronArray[] input, int inputStart, int inputEnd, DenseNeuronArray output, int outputStart, int outputEnd, OutputMappingSparseArrayToDense mapping);

}
