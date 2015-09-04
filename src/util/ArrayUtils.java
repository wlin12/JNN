package util;

public class ArrayUtils {
	public static void concat(Object[]a, Object[] b, Object[] ret){
		for(int i = 0; i < a.length; i++){
			ret[i] = a[i];
		}
		for(int i = 0; i < b.length; i++){
			ret[i + a.length] = b[i];
		}
	}
	
	public static int[] concat(int[]a, int[] b){
		int[] ret = new int[a.length + b.length];
		for(int i = 0; i < a.length; i++){
			ret[i] = a[i];
		}
		for(int i = 0; i < b.length; i++){
			ret[i + a.length] = b[i];
		}
		return ret;
	}
	
	public static String[] concat(String[]a, String[] b){
		String[] ret = new String[a.length + b.length];
		for(int i = 0; i < a.length; i++){
			ret[i] = a[i];
		}
		for(int i = 0; i < b.length; i++){
			ret[i + a.length] = b[i];
		}
		return ret;
	}
}
