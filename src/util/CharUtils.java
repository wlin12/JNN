package util;

public class CharUtils {


	public static boolean isASCII(String word){
		for(int i = 0; i < word.length(); i++){
			if(word.charAt(i) >= 256){
				return false;
			}
		}
		return true;
	}
	
}
