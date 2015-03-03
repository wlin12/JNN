package jnn.functions;

import jnn.mapping.OutputMappingSparseToSparse;
import jnn.neuron.SparseNeuronArray;

public interface SparseToSparseTransform {
	public void forward(SparseNeuronArray input, int inputStart, int inputEnd, SparseNeuronArray output, int outputStart, int outputEnd, OutputMappingSparseToSparse mapping);
	public void backward(SparseNeuronArray input, int inputStart, int inputEnd, SparseNeuronArray output, int outputStart, int outputEnd, OutputMappingSparseToSparse mapping);

}
