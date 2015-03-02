package util;

public class StringUtils {
	public static String arrayToString(String[] array){
		String ret = "";
		for(String s : array){
			ret+=s+"_";
		}
		return ret.substring(0, ret.length()-1);
	}
	
	public static String arrayToString(double[] array){
		String ret = "";
		for(double s : array){
			ret+=s+"_";
		}
		return ret.substring(0, ret.length()-1);
	}
	
	public static String arrayToString(double[] array, String delimiter){
		String ret = "";
		for(double s : array){
			ret+=s+delimiter;
		}
		return ret.substring(0, ret.length()-delimiter.length());
	}

}
