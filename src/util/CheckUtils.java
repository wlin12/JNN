package util;

public class CheckUtils {

	public static void checkGT(int a, int b){
		if(a > b){
			return;
		}
		else{
			System.err.println("Error: " + a + " le than " + b);
			System.exit(0);
		}
	}
	
	public static void checkEQ(int a, int b){
		if(a == b){
			return;
		}
		else{
			System.err.println("Error: " + a + " not eq to " + b);
			System.exit(0);
		}
	}

	public static void checkMXSize(double[][] a, double[][] b){
		if(a.length == b.length){
			if(a[0].length == b[0].length){
				return;
			}
			else{				
				System.err.println("Error: dim 2 - " + a[0].length + " not eq to " + b[0].length);
				System.exit(0);
			}
		}
		else{
			System.err.println("Error: dim 1 - " + a.length + " not eq to " + b.length);
			System.exit(0);
		}
	}
}
