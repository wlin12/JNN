package jnn.mapping;

import jnn.functions.DenseToSparseTransform;
import jnn.functions.parametrized.Layer;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.NeuronArray;
import jnn.neuron.SparseNeuronArray;

public class OutputMappingDenseToSparse extends Mapping{

	DenseNeuronArray input;
	SparseNeuronArray output;
	DenseToSparseTransform layer;
		
	public OutputMappingDenseToSparse(int inputStart, int inputEnd,
			int outputStart, int outputEnd, DenseNeuronArray input,
			SparseNeuronArray output, DenseToSparseTransform layer) {
		super(inputStart, inputEnd, outputStart, outputEnd);
		this.input = input;
		this.output = output;
		this.layer = layer;
		validate();
		setTimedLayer(layer);
	}
	
	public OutputMappingDenseToSparse(DenseNeuronArray input,
			SparseNeuronArray output, DenseToSparseTransform layer) {
		super(0, input.size-1, 0, output.size-1);
		this.input = input;
		this.output = output;
		this.layer = layer;
		validate();
	}
	
	@Override
	public void forward() {
		long time = System.currentTimeMillis();

		layer.forward(input, inputStart, inputEnd, output, outputStart, outputEnd, this);
		((Layer)layer).addForward(System.currentTimeMillis()-time);
	}
	
	@Override
	public void backward() {
		long time = System.currentTimeMillis();
		layer.backward(input, inputStart, inputEnd, output, outputStart, outputEnd, this);	
		((Layer)layer).addBackward(System.currentTimeMillis()-time);

	}
	
	@Override
	public Layer getLayer() {
		return (Layer)layer;
	}
	
	public DenseNeuronArray getInput() {
		return input;
	}
	
	public SparseNeuronArray getOutput() {
		return output;
	}
	
	@Override
	public NeuronArray[] getInputArray() {
		return new DenseNeuronArray[]{input};
	}
	
	@Override
	public NeuronArray[] getOutputArray() {
		return new SparseNeuronArray[]{output};
	}
	
	@Override
	public String toString() {
		String ret = "from: " + input + "\n";
		ret += "to: " + output + "\n";
		ret += "indexes : " + inputStart + ":" + inputEnd + " -> " + outputStart + ":"+outputEnd; 
		return ret;
	}
}
