package jnn.functions.nlp.aux.input;

public class InputSentence{
	public String[] tokens;
	public String original;

	public InputSentence(String original, String[] tokens) {
		super();
		this.original = original;
		this.tokens = tokens;
	}
	
	public String[] getTokens() {
		return tokens;
	}
}
