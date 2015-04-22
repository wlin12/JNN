package jnn.neuron;

abstract public class NeuronArray {
	public int size;

	public NeuronArray(int size) {
		super();
		this.size = size;
	}
	
	public void beforeForward() {}

	public void beforeBackward() {}

	abstract public void init();
	abstract public void capValues();

}
