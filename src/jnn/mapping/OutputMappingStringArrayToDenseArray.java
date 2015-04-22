package jnn.mapping;

import jnn.functions.StringArrayToDenseArrayTransform;
import jnn.functions.StringToDenseTransform;
import jnn.functions.parametrized.Layer;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.NeuronArray;

public class OutputMappingStringArrayToDenseArray extends Mapping{
	String[] input;
	DenseNeuronArray[] output;
	StringArrayToDenseArrayTransform layer;

	public OutputMappingStringArrayToDenseArray(String[] input,
			int outputStart, int outputEnd,
			DenseNeuronArray[] output, StringArrayToDenseArrayTransform layer) {
		super(0, 0, outputStart, outputEnd);		
		this.output = output;
		this.layer = layer;
		this.input = input;
		setTimedLayer(layer);
	}
	
	public OutputMappingStringArrayToDenseArray(String[] input, DenseNeuronArray[] output, StringArrayToDenseArrayTransform layer) {
		super(0, 0, 0, output[0].size-1);
		this.output = output;
		this.layer = layer;
		this.input = input;
	}

	@Override
	public void forward() {				
		long time = System.currentTimeMillis();
		layer.forward(input,output, outputStart, outputEnd, this);
		((Layer)layer).addForward(System.currentTimeMillis()-time);
	}

	@Override
	public void backward() {
		long time = System.currentTimeMillis();
		layer.backward(input, output, outputStart, outputEnd, this);
		((Layer)layer).addBackward(System.currentTimeMillis()-time);
	}	
	
	@Override
	public Layer getLayer() {		
		return (Layer)layer;
	}
	
	@Override
	public String toString() {			
		String ret = "from:\n";
		for(int i = 0 ; i < input.length; i++){
			ret+=input[i]+"\n";
		}
		ret += "to:\n";
		for(int i = 0 ; i < output.length; i++){
			ret+=output[i]+"\n";
		}
		ret += "indexes : " + inputStart + ":" + inputEnd + " -> " + outputStart + ":"+outputEnd; 
		return ret;
	}
	
	public DenseNeuronArray getInput() {
		return null;
	}
	
	public DenseNeuronArray getOutput() {
		return output[0];
	}
	
	@Override
	public NeuronArray[] getInputArray() {
		return null;
	}
	
	@Override
	public NeuronArray[] getOutputArray() {
		return output;
	}
}
