package jnn.mapping;

import jnn.functions.StringArrayToDenseArrayTransform;
import jnn.functions.StringArrayToStringArrayTransform;
import jnn.functions.parametrized.Layer;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.NeuronArray;
import jnn.neuron.StringNeuronArray;

public class OutputMappingStringArrayToStringArray extends Mapping{
	String[] input;
	StringNeuronArray[] output;
	StringArrayToStringArrayTransform layer;

	public OutputMappingStringArrayToStringArray(String[] input,
			StringNeuronArray[] output, StringArrayToStringArrayTransform layer) {
		super(0, 0, 0, 0);		
		this.output = output;
		this.layer = layer;
		this.input = input;
		setTimedLayer(layer);
	}
	
	@Override
	public void forward() {				
		long time = System.currentTimeMillis();
		layer.forward(input,output, this);
		((Layer)layer).addForward(System.currentTimeMillis()-time);
	}

	@Override
	public void backward() {
		long time = System.currentTimeMillis();
		layer.backward(input, output, this);
		((Layer)layer).addBackward(System.currentTimeMillis()-time);
	}	
	
	@Override
	public Layer getLayer() {		
		return (Layer)layer;
	}
	
	@Override
	public String toString() {
		String ret = "to: " + output + "\n";
		for(int i = 0; i < output.length; i++){
			ret+=output[i]+"\n";
		}
		ret += "indexes : " + inputStart + ":" + inputEnd + " -> " + outputStart + ":"+outputEnd; 
		return ret;
	}
	
	public NeuronArray getInput() {
		return null;
	}
	
	public NeuronArray getOutput() {
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
