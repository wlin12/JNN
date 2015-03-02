package jnn.neuron;

public class NeuronArray {
	public int size;

	public NeuronArray(int size) {
		super();
		this.size = size;
	}
	
	public void init(){
		throw new RuntimeException("not implemented, please overload this method");
	}

	public void beforeForward() {}

	public void beforeBackward() {}

	public void capValues() {
		throw new RuntimeException("not implemented, please overload this method");		
	};

}
