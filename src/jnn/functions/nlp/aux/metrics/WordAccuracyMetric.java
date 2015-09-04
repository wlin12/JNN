package jnn.functions.nlp.aux.metrics;

import java.io.PrintStream;

import jnn.functions.nlp.aux.input.LabelledSentence;
import jnn.objective.SentenceSoftmaxDenseObjective;

public class WordAccuracyMetric extends WordBasedEvalMetric{

	ErrorStats error = new ErrorStats();
	
	public WordAccuracyMetric(String title) {
		super(title);
		error.init();
		error.setReverse();
	}
	
	@Override
	public void addSentenceScore(LabelledSentence sentence, LabelledSentence hypothesis) {
		for(int i = 0; i < sentence.tokens.length; i++){
			if(!sentence.tags[i].equals(hypothesis.tags[i])){
				error.addError(1);				
			}
			else{
				error.addError(0);				
			}
		}
	}
	
	@Override
	public void commit() {
		error.commitError();		
		error.initError();
	}
	
	@Override
	public void print(PrintStream out) {
		error.displayResults(title, out);
	}
	
	@Override
	public void printShort(PrintStream out) {
		error.displayResults(title, out, true);
	}
	
	@Override
	public boolean isBestIteration() {
		return error.isBestIteration();
	}
	
	@Override
	public int getBestIteration() {
		return error.minErrorIteration;
	}
	
	@Override
	public double getScoreAtIteration(int iteration) {
		return error.getScoreAtIteration(iteration);
	}
}
