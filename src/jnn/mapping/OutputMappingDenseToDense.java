package jnn.mapping;

import jnn.functions.DenseToDenseTransform;
import jnn.functions.parametrized.Layer;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.NeuronArray;

public class OutputMappingDenseToDense extends Mapping{
	DenseNeuronArray input;
	DenseNeuronArray output;
	DenseToDenseTransform layer;	

	public OutputMappingDenseToDense(int inputStart, int inputEnd,
			int outputStart, int outputEnd, DenseNeuronArray input,
			DenseNeuronArray output, DenseToDenseTransform layer) {
		super(inputStart, inputEnd, outputStart, outputEnd);
		this.input = input;
		this.output = output;
		this.layer = layer;
		validate();
		setTimedLayer(layer);
	}
	
	public OutputMappingDenseToDense(DenseNeuronArray input,
			DenseNeuronArray output, DenseToDenseTransform layer) {
		super(0, input.size-1, 0, output.size-1);
		this.input = input;
		this.output = output;
		this.layer = layer;
		validate();
	}

	@Override
	public void forward() {
		if(!input.isOutputInitialized()){
			throw new RuntimeException("input is null");
		}
		if(!output.isOutputInitialized()){
			throw new RuntimeException("output is null");
		}
		long time = System.currentTimeMillis();
		try{
			layer.forward(input, inputStart, inputEnd, output, outputStart, outputEnd, this);
		} catch (ArrayIndexOutOfBoundsException e){
			System.err.println("error mapping using layer " + layer);
			System.err.println("input was " + input);
			System.err.println("output was " + output);
			throw (e);
		}
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
	
	public DenseNeuronArray getInput() {
		return input;
	}
	
	public DenseNeuronArray getOutput() {
		return output;
	}
	
	@Override
	public NeuronArray[] getInputArray() {
		return new DenseNeuronArray[]{input};
	}
	
	@Override
	public NeuronArray[] getOutputArray() {
		return new DenseNeuronArray[]{output};
	}
}
