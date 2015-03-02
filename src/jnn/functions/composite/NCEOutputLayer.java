package jnn.functions.composite;
import jnn.functions.DenseArrayToDenseTransform;
import jnn.functions.DenseToSparseTransform;
import jnn.functions.parametrized.DenseFullyConnectedLayer;
import jnn.functions.parametrized.Layer;
import jnn.functions.parametrized.SparseOutputFullyConnectedLayer;
import jnn.mapping.OutputMappingDenseArrayToDense;
import jnn.mapping.OutputMappingDenseToSparse;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.SparseNeuronArray;

public class NCEOutputLayer extends Layer implements DenseToSparseTransform, DenseArrayToDenseTransform{
	int inputDim;
	int outputDim;
	
	DenseFullyConnectedLayer denseOutput;
	SparseOutputFullyConnectedLayer sparseOutput;
	
	public NCEOutputLayer(int inputDim, int outputDim) {
		super();
		this.inputDim = inputDim;
		this.outputDim = outputDim;
		sparseOutput = new SparseOutputFullyConnectedLayer(inputDim, outputDim);
		denseOutput = new DenseFullyConnectedLayer(inputDim, outputDim);
	}

	@Override
	public void forward(DenseNeuronArray[] input, int inputStart, int inputEnd,
			DenseNeuronArray output, int outputStart, int outputEnd,
			OutputMappingDenseArrayToDense mapping) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void backward(DenseNeuronArray[] input, int inputStart,
			int inputEnd, DenseNeuronArray output, int outputStart,
			int outputEnd, OutputMappingDenseArrayToDense mapping) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void forward(DenseNeuronArray input, int inputStart, int inputEnd,
			SparseNeuronArray output, int outputStart, int outputEnd,
			OutputMappingDenseToSparse mapping) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void backward(DenseNeuronArray input, int inputStart, int inputEnd,
			SparseNeuronArray output, int outputStart, int outputEnd,
			OutputMappingDenseToSparse mapping) {
		// TODO Auto-generated method stub
		
	}
	
	
}
