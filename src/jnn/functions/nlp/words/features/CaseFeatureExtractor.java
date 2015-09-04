package jnn.functions.nlp.words.features;

import util.LangUtils;

public class CaseFeatureExtractor implements FeatureExtractor{
	@Override
	public String extract(String word) {
		if(word.length() == 0) return "none";
		if(Character.isUpperCase(word.charAt(0))) return "capitalized";
		if(LangUtils.isUppercased(word)) return "uppercased";
		return "lowercase";
	}

}
