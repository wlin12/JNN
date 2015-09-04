package jnn.functions.nlp.words.features;

import util.LangUtils;

public class CharSequenceExtractorLc implements SequenceExtractor{
	@Override
	public String[] extract(String word) {
		return LangUtils.splitWord(word);
	}
}
