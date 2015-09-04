package jnn.functions.nlp.words.features;

import util.LangUtils;

public class CharSequenceExtractor implements SequenceExtractor{
	@Override
	public String[] extract(String word) {
		return LangUtils.splitWord(word);
	}
}
