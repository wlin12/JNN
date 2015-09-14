package jnn.decoder.stackbased;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

import jnn.decoder.DecoderInterface;
import jnn.decoder.DecoderStack;
import jnn.decoder.state.DecoderState;

public class StackBasedDecoder {
	
	//prerequisites
	public DecoderInterface decoderMethods;	
	public DecoderState initialState;

	//parameters
	public int stackSize = 20;
	public int maxTimestamp = 200;
	public double maxMargin = 0.1;
	
	//local vars
	public HashMap<Integer,DecoderStack> stacksByTimestamp = new HashMap<Integer, DecoderStack>();
	DecoderStack finalStates = new DecoderStack();
	
	public StackBasedDecoder(DecoderInterface decoderMethods, DecoderState initialState) {
		super();
		this.decoderMethods = decoderMethods;
		this.initialState = initialState;
	}
	
	public void decode(){
		double bestFinalStateScore = -Double.MAX_VALUE;		
		if(stacksByTimestamp.isEmpty()){
			getStack(0).addState(initialState);
			for(int t = 0; t < maxTimestamp; t++){
				DecoderStack stack = getStack(t);
				for(int s = 0; s < stackSize; s++){
					DecoderState state = stack.popState();					
					if(state == null) break;
					if(state.score < bestFinalStateScore) continue;
					List<DecoderState> followingStates = decoderMethods.expand(state);
					for(DecoderState next : followingStates){
						next.setPrevState(state);
						if(next.isFinal){
							finalStates.addState(next);
							if(bestFinalStateScore < next.score){
								bestFinalStateScore = next.score;
							}
						}
						else{
							getStack(next.size + t).addState(next);
						}
					}
				}
				stack.removeAll();
			}
		}
	}
	
	public DecoderState getBestState(){
		return finalStates.peekTop();
	}
	
	public LinkedList<DecoderState> getBestStates(int topN){
		LinkedList<DecoderState> ret = new LinkedList<DecoderState>();
		while(finalStates.peekTop() != null && ret.size() < topN){
			ret.addLast(finalStates.peekTop());
			finalStates.popState();
		}
		return ret;
	}
	
	public DecoderStack getStack(int timestamp){
		if(!stacksByTimestamp.containsKey(timestamp)){
			stacksByTimestamp.put(timestamp, new DecoderStack());
		}
		return stacksByTimestamp.get(timestamp);
	}
	
	public void addFinalState(DecoderState state){
		finalStates.addState(state);
	}
}
