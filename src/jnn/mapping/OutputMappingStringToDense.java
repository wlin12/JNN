package jnn.mapping;

import jnn.functions.StringToDenseTransform;
import jnn.functions.parametrized.Layer;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.NeuronArray;

public class OutputMappingStringToDense extends Mapping{
	String input;
	DenseNeuronArray output;
	StringToDenseTransform layer;

	public OutputMappingStringToDense(String input,
			int outputStart, int outputEnd,
			DenseNeuronArray output, StringToDenseTransform layer) {
		super(0, 0, outputStart, outputEnd);		
		this.output = output;
		this.layer = layer;
		this.input = input;
		setTimedLayer(layer);
	}
	
	public OutputMappingStringToDense(String input, DenseNeuronArray output, StringToDenseTransform layer) {
		super(0, 0, 0, output.size-1);
		this.output = output;
		this.layer = layer;
		this.input = input;
	}

	@Override
	public void forward() {		
		if(!output.isOutputInitialized()){
			throw new RuntimeException("output is null");
		}
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
		String ret = "to: " + output + "\n";
		ret += "indexes : " + inputStart + ":" + inputEnd + " -> " + outputStart + ":"+outputEnd; 
		return ret;
	}
	
	public DenseNeuronArray getInput() {
		return null;
	}
	
	public DenseNeuronArray getOutput() {
		return output;
	}
	
	@Override
	public NeuronArray[] getInputArray() {
		return null;
	}
	
	@Override
	public NeuronArray[] getOutputArray() {
		return new DenseNeuronArray[]{output};
	}
}
