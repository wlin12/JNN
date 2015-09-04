package jnn.functions.nlp.aux.metrics;

import java.io.PrintStream;
import java.util.ArrayList;

import jnn.functions.nlp.aux.input.LabelledSentence;

abstract public class WordBasedEvalMetric {
	String title;
	abstract public void addSentenceScore(LabelledSentence sentence, LabelledSentence hypothesis);
	abstract public void commit();
	abstract public void print(PrintStream out);
	abstract public boolean isBestIteration();

	public void printShort(PrintStream out){
		print(out);
	}

	public WordBasedEvalMetric(String title) {
		super();
		this.title = title;
	}
	
	public void addSentenceScore(ArrayList<LabelledSentence> sents, ArrayList<LabelledSentence> hypothesis){
		for(int i = 0; i < sents.size(); i++){
			addSentenceScore(sents.get(i), hypothesis.get(i));
		}
	}
	
	abstract public int getBestIteration();
	
	abstract public double getScoreAtIteration(int iteration);
}
