package jnn.functions;

import jnn.mapping.OutputMappingSparseToDense;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.SparseNeuronArray;

public interface SparseToDenseTransform {
	public void forward(SparseNeuronArray input, int inputStart, int inputEnd, DenseNeuronArray output, int outputStart, int outputEnd, OutputMappingSparseToDense mapping);
	public void backward(SparseNeuronArray input, int inputStart, int inputEnd, DenseNeuronArray output, int outputStart, int outputEnd, OutputMappingSparseToDense mapping);

}
