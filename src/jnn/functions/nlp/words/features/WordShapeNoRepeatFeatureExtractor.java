package jnn.functions.nlp.words.features;

import util.LangUtils;

public class WordShapeNoRepeatFeatureExtractor implements FeatureExtractor{
	@Override
	public String extract(String word) {	
		return word.replaceAll("[a-z]", "x").replaceAll("[0-9]", "d").replaceAll("[A-Z]","X");
	}
}
