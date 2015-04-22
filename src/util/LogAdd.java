package util;

/**
 * Method to provide the static function logAdd which is useful when arithmetic
 * has to be done on the log scale.
 * @author nicholasbartlett
 */
public class LogAdd {
	/**
	 * Method to get the log(A + B) when provided log(A) and log(B).
	 *
	 * @param logA log(A)
	 * @param logB log(B)
	 * @return log(A + B)
	 */
	@SuppressWarnings("FinalStaticMethod")
	public final static double logAdd(double logA, double logB){
		if(Double.isInfinite(logA) && Double.isInfinite(logB) && logA < 0 && logB < 0){
			return Double.NEGATIVE_INFINITY;
		} else if(logA > logB){
			logB -= logA;
			return logA + Math.log(1.0 + Math.exp(logB));
		} else {
			logA -= logB;
			return logB + Math.log(Math.exp(logA) + 1.0);
		}
	}
}