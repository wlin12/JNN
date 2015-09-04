package jnn.functions.nlp.words.features;

public class CapitalizationWordFeatureExtractor implements FeatureExtractor{
	@Override
	public String extract(String word) {
		if(word.length() == 0) return "false";
		if(Character.isUpperCase(word.charAt(0))) return "true";
		return "false";
	}
}
