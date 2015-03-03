package jnn.functions;

import jnn.mapping.OutputMappingDenseToSparse;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.SparseNeuronArray;

public interface DenseToSparseTransform {
	public void forward(DenseNeuronArray input, int inputStart, int inputEnd, SparseNeuronArray output, int outputStart, int outputEnd, OutputMappingDenseToSparse mapping);
	public void backward(DenseNeuronArray input, int inputStart, int inputEnd, SparseNeuronArray output, int outputStart, int outputEnd, OutputMappingDenseToSparse mapping);

}
