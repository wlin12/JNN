package jnn.functions.nlp.aux.readers;

import jnn.functions.nlp.aux.input.InputSentence;


public class StanfordFileReader implements TextReader{
	public InputSentence readSentence(String input) {
		return StanfordPosReader.lineToPosTaggedSentence(input);
	};
}
