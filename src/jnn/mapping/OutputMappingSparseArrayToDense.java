package jnn.mapping;

import jnn.functions.SparseArrayToDenseTransform;
import jnn.functions.parametrized.Layer;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.NeuronArray;
import jnn.neuron.SparseNeuronArray;

public class OutputMappingSparseArrayToDense extends Mapping{
	SparseNeuronArray[] input;
	DenseNeuronArray output;
	SparseArrayToDenseTransform layer;

	public OutputMappingSparseArrayToDense(int inputStart, int inputEnd,
			int outputStart, int outputEnd, SparseNeuronArray[] input,
			DenseNeuronArray output, SparseArrayToDenseTransform layer) {
		super(inputStart, inputEnd, outputStart, outputEnd);
		this.input = input;
		this.output = output;
		this.layer = layer;
		validate();
		setTimedLayer(layer);
	}	
	
	public OutputMappingSparseArrayToDense(SparseNeuronArray input[],
			DenseNeuronArray output, SparseArrayToDenseTransform layer) {
		super(0, input[0].size-1, 0, output.size-1);		
		this.input = input;
		this.output = output;
		this.layer = layer;
		validate();
	}
	
	public void validate(){
		super.validate();
		
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
	
	@Override
	public String toString() {
		String ret = "from: " + input + "\n";
		ret += "to: " + output + "\n";
		ret += "indexes : " + inputStart + ":" + inputEnd + " -> " + outputStart + ":"+outputEnd; 
		return ret;
	}
	
	public SparseNeuronArray getInput() {
		return input[0];
	}
	
	public DenseNeuronArray getOutput() {
		return output;
	}
	
	@Override
	public NeuronArray[] getInputArray() {
		return input;
	}
	
	@Override
	public NeuronArray[] getOutputArray() {
		return new DenseNeuronArray[]{output};
	}
}
