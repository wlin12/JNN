package jnn.functions.nlp.aux.readers;

import java.util.List;

import jnn.functions.nlp.aux.input.InputSentence;
import util.twitter.Twokenize;

public class TwokenizerFileReader implements TextReader{
	public InputSentence readSentence(String input) {
		List<String> ret = Twokenize.tokenize(input);
		String[] tokens = new String[ret.size()];
		int i = 0;
		for(String s : ret){
			tokens[i++] = s;
		}
		return new InputSentence(input, tokens);
	}
}
