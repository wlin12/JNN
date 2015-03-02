package util;

import java.util.Random;

/**
 * A subclass of java.util.random that implements the 
 * Xorshift random number generator
 */

public class XorRandom extends Random {
 private long seed;

 public XorRandom(long seed) {
  this.seed = seed;
 }

 protected int next(int nbits) {
  long x = seed;
  x ^= (x << 21);
  x ^= (x >>> 35);
  x ^= (x << 4);
  seed = x;
  x &= ((1L << nbits) - 1);
  return (int) x;
 }
}