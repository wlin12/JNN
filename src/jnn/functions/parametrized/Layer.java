package jnn.functions.parametrized;

import jnn.training.GraphInference;

abstract public class Layer {

	public long commitTime = 0;
	public long commits = 0;
	public long forwardTime = 0;
	public long forwardCalls = 0;
	public long backwardTime = 0;
	public long backwardCalls = 0;
	
	public void updateWeights(double learningRate, double momentum){
		
	}
	
	public void updateWeightsTimed(double learningRate, double momentum){
		long before = System.currentTimeMillis();
		updateWeights(learningRate, momentum);
		long after = System.currentTimeMillis();
		addCommit(after-before);
	}
	
	public void reset(){
		
	}
	
	public void addCommit(long timeinmillis){
		commitTime+=timeinmillis;
		commits++;
	}
	
	public void addForward(long timeinmillis){
		forwardTime+=timeinmillis;
		forwardCalls++;
	}
	
	public void addBackward(long timeinmillis){
		backwardTime+=timeinmillis;
		backwardCalls++;
	}
	
	public void backwardEnd(GraphInference inference){
		
	}
	
	public void printCommitTimeAndReset(){
		if(commits>0){
			System.err.println("took " + commitTime + " milliseconds to commit: " + commitTime/commits + " miliseconds per commit (" + commits + ")");			
		}
		else{
			System.err.println("no commit made... ");						
		}
		commitTime=0;
		commits=0;
		if(forwardCalls>0){
			System.err.println("took " + forwardTime + " milliseconds to forward: " + forwardTime/forwardCalls + " miliseconds per forward (" + forwardCalls + ")");			
		}
		else{
			System.err.println("no forward made... ");						
		}
		forwardTime=0;
		forwardCalls=0;
		if(backwardCalls>0){
			System.err.println("took " + backwardTime + " milliseconds to backward: " + backwardTime/backwardCalls + " miliseconds per backward (" + backwardCalls + ")");			
		}
		else{
			System.err.println("no backward made... ");						
		}
		backwardTime=0;
		backwardCalls=0;
	}
	
}
