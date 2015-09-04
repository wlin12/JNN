package jnn.functions.nlp.words.features;

import util.LangUtils;

public class WordShapeFeatureExtractor implements FeatureExtractor{
	@Override
	public String extract(String word) {
		return LangUtils.noRepeat(word.replaceAll("[a-z]", "x").replaceAll("[0-9]", "d").replaceAll("[A-Z]","X"));
	}
}
