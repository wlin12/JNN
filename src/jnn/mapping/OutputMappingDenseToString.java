package jnn.mapping;

import jnn.functions.DenseToStringTransform;
import jnn.functions.parametrized.Layer;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.NeuronArray;
import jnn.neuron.StringNeuronArray;

public class OutputMappingDenseToString extends Mapping{
	DenseNeuronArray input;
	StringNeuronArray output;
	DenseToStringTransform layer;

	public OutputMappingDenseToString(DenseNeuronArray input,
			int inputStart, int inputEnd, StringNeuronArray output, DenseToStringTransform layer) {
		super(inputStart, inputEnd, 0 ,0);		
		this.output = output;
		this.layer = layer;
		this.input = input;
		setTimedLayer(layer);
	}
	
	public OutputMappingDenseToString(DenseNeuronArray input, StringNeuronArray output, DenseToStringTransform layer) {
		super(0, input.size-1, 0, 0);
		this.output = output;
		this.layer = layer;
		this.input = input;
	}

	@Override
	public void forward() {		
		if(!input.isOutputInitialized()){
			throw new RuntimeException("input is null");
		}
		long time = System.currentTimeMillis();
		layer.forward(input,inputStart,inputEnd,output, this);
		((Layer)layer).addForward(System.currentTimeMillis()-time);
	}

	@Override
	public void backward() {
		long time = System.currentTimeMillis();
		layer.backward(input, inputStart,inputEnd, output, this);
		((Layer)layer).addBackward(System.currentTimeMillis()-time);
	}	
	
	@Override
	public Layer getLayer() {		
		return (Layer)layer;
	}
	
	@Override
	public String toString() {
		String ret = "to: " + output + "\n";
		ret += "indexes : " + inputStart + ":" + inputEnd + " -> " + outputStart + ":"+outputEnd; 
		return ret;
	}
	
	public DenseNeuronArray getInput() {
		return input;
	}
	
	public DenseNeuronArray getOutput() {
		return null;
	}
	
	@Override
	public NeuronArray[] getInputArray() {
		return new DenseNeuronArray[]{input};
	}
	
	@Override
	public NeuronArray[] getOutputArray() {
		return null;
	}
}
