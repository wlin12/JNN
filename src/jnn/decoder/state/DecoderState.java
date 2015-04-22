package jnn.decoder.state;


public class DecoderState {
	public String name;
	public double score;
	public boolean isFinal; // do not expand final states and are candidates for output
	public int size = 1; // number of timestamps to advance
	public DecoderState prevState;
	public int numberOfPrevStates = 0;

	public DecoderState(double score, boolean isFinal) {
		super();
		this.score = score;
		this.isFinal = isFinal;
	}

	public DecoderState(String name, double score, boolean isFinal) {
		super();
		this.name = name;
		this.score = score;
		this.isFinal = isFinal;
	}	
	
	@Override
	public String toString() {
		return "name:" + name + ", score:" + score;
	}
	
	public void setSize(int size) {
		this.size = size;
	}
	
	public void setPrevState(DecoderState prevState) {
		this.prevState = prevState;
		this.numberOfPrevStates = prevState.numberOfPrevStates+1;
	}
}
