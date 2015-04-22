package jnn.decoder;

import jnn.decoder.state.DecoderState;
import external.FibonacciHeap;


public class DecoderStack {
	FibonacciHeap<DecoderState> states = new FibonacciHeap<DecoderState>();
	
	public void addState(DecoderState state){
		states.enqueue(state, -state.score);
	}
	
	public DecoderState popState(){
		if(states.size()==0) return null;
		return states.dequeueMin().getValue();
	}

	public DecoderState peekTop() {
		if(states.size()==0) return null;
		return states.min().getValue();		
	}
	
	public void removeAll() {
		states = new FibonacciHeap<DecoderState>();
	}
	
	public static void main(String[] args){
		DecoderStack stack = new DecoderStack();
		stack.addState(new DecoderState("buck",0.1,false));
		stack.addState(new DecoderState("peter",0.2,false));
		stack.addState(new DecoderState("mary",0.3, false));
		stack.addState(new DecoderState("john",0.4, false));
		System.err.println(stack.states.min().getValue());
		stack.states.dequeueMin();
		System.err.println(stack.states.min().getValue());
	}

	

}
