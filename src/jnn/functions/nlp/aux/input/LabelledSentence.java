package jnn.functions.nlp.aux.input;

public class LabelledSentence extends InputSentence{
	public String[] tags;
	
	public LabelledSentence(String original, String[] tokens, String[] tags) {
		super(original, tokens);
		this.tags = tags;
	}
	
	public String[] getTags() {
		return tags;
	}
	
	@Override
	public String toString() {
		String ret = "";
		for(int i = 0; i < tags.length; i++){
			ret+= tokens[i] + "(" + tags[i] + ") ";
		}
		return ret;
	}
	
	public String toStanford(){
		String ret = "";
		for(int i = 0; i < tags.length; i++){
			ret+= tokens[i] + "_" + tags[i] + " ";
		}
		return ret.trim();
	}

	public String toText() {
		String ret = "";
		for(int i = 0; i < tags.length; i++){
			ret+= tokens[i] + " ";
		}
		return ret.trim();
	}
}
