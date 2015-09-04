package jnn.functions.nlp.aux.readers;

import java.util.List;

import jnn.functions.nlp.aux.input.InputSentence;
import util.twitter.Twokenize;

public class PlainTextReader implements TextReader{
	public InputSentence readSentence(String input) {
		String[] tokens = input.split("\\s+");
		return new InputSentence(input, tokens);
	}
}
