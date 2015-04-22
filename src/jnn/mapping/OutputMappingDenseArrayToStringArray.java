package jnn.mapping;

import jnn.functions.DenseArrayToStringArrayTransform;
import jnn.functions.DenseToStringTransform;
import jnn.functions.parametrized.Layer;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.NeuronArray;
import jnn.neuron.StringNeuronArray;

public class OutputMappingDenseArrayToStringArray extends Mapping{
	DenseNeuronArray[] input;
	StringNeuronArray[] output;
	DenseArrayToStringArrayTransform layer;

	public OutputMappingDenseArrayToStringArray(DenseNeuronArray[] input,
			int inputStart, int inputEnd, StringNeuronArray[] output, DenseArrayToStringArrayTransform layer) {
		super(inputStart, inputEnd, 0 ,0);		
		this.output = output;
		this.layer = layer;
		this.input = input;
		setTimedLayer(layer);
	}
	
	public OutputMappingDenseArrayToStringArray(DenseNeuronArray[] input, StringNeuronArray[] output, DenseArrayToStringArrayTransform layer) {
		super(0, input[0].size-1, 0, 0);
		this.output = output;
		this.layer = layer;
		this.input = input;
	}

	@Override
	public void forward() {		
		for(DenseNeuronArray i : input){
			if(!i.isOutputInitialized()){
				throw new RuntimeException("input is null");
			}
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
	
	public NeuronArray getInput() {
		return input[0];
	}
	
	public NeuronArray getOutput() {
		return output[0];
	}
	
	@Override
	public NeuronArray[] getInputArray() {
		return input;
	}
	
	@Override
	public NeuronArray[] getOutputArray() {
		return output;
	}
}
