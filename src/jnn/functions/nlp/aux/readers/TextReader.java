package jnn.functions.nlp.aux.readers;

import jnn.functions.nlp.aux.input.InputSentence;


public interface TextReader {
	public InputSentence readSentence(String input);

}
