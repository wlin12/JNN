package jnn.mapping;

import jnn.functions.SparseToDenseTransform;
import jnn.functions.parametrized.Layer;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.NeuronArray;
import jnn.neuron.SparseNeuronArray;

public class OutputMappingSparseToDense extends Mapping{
	SparseNeuronArray input;
	DenseNeuronArray output;
	SparseToDenseTransform layer;

	public OutputMappingSparseToDense(int inputStart, int inputEnd,
			int outputStart, int outputEnd, SparseNeuronArray input,
			DenseNeuronArray output, SparseToDenseTransform layer) {
		super(inputStart, inputEnd, outputStart, outputEnd);
		this.input = input;
		this.output = output;
		this.layer = layer;
		validate();
		setTimedLayer(layer);
	}	
	
	public OutputMappingSparseToDense(SparseNeuronArray input,
			DenseNeuronArray output, SparseToDenseTransform layer) {
		super(0, input.size-1, 0, output.size-1);		
		if(layer == null) throw new RuntimeException("layer is null");
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
		if(layer == null){
			System.err.println(this);
			throw new RuntimeException("layer is null ");
		}
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
		return input;
	}
	
	public DenseNeuronArray getOutput() {
		return output;
	}
	
	@Override
	public NeuronArray[] getInputArray() {
		return new SparseNeuronArray[]{input};
	}
	
	@Override
	public NeuronArray[] getOutputArray() {
		return new DenseNeuronArray[]{output};
	}
}
