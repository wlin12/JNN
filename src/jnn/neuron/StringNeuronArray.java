package jnn.neuron;

public class StringNeuronArray extends NeuronArray{
	public String output;
	public String expected;
	public double score;
	public boolean negative = false;
	
	public StringNeuronArray() {
		super(1);
	}	
	
	@Override
	public void init() {
		output = "";
	}
	
	@Override
	public void capValues() {
		
	}	
	
	public String getOutput() {
		return output;
	}
	
	public String getExpected() {
		return expected;
	}
	
	public void setExpected(String expected) {
		this.expected = expected;
	}
	
	public void setOutput(String input) {
		this.output = input;
	}

	public static StringNeuronArray[] asArray(int length) {
		StringNeuronArray[] ret = new StringNeuronArray[length];
		for(int i = 0; i < ret.length; i++){
			ret[i] = new StringNeuronArray();
		}
		return ret;
	}
	
	public static void setExpectedArray(String[] expected, StringNeuronArray[] toSet){
		for(int i = 0; i < expected.length; i++){
			toSet[i].setExpected(expected[i]);
		}
	}
	
	@Override
	public String toString() {
		String ret = "output: " + output;
		if(expected != null){
			ret += " expected: " + expected;
		}
		return ret;
	}

	public static String[] toStringArray(StringNeuronArray[] predictions) {
		String[] ret = new String[predictions.length];
		for(int i = 0; i < ret.length; i++){
			ret[i] = predictions[i].getOutput();
		}
		return ret;
	}
	
	public static String[] toStringArrayExpected(StringNeuronArray[] predictions) {
		String[] ret = new String[predictions.length];
		for(int i = 0; i < ret.length; i++){
			ret[i] = predictions[i].getExpected();
		}
		return ret;
	}

	public void setScore(double score) {
		this.score = score;
	}
	
	public double getScore() {
		return score;
	}

	public static void setNegative(StringNeuronArray[] outputNeurons) {
		for(int i = 0; i < outputNeurons.length; i++){
			outputNeurons[i].negative = true;
		}
	}
}
