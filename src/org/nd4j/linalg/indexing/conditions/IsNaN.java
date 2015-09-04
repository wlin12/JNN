package org.nd4j.linalg.indexing.conditions;

import org.nd4j.linalg.api.complex.IComplexNumber;

/**
 * Condition for whether an element is NaN
 * @author Adam Gibson
 */
public class IsNaN implements Condition {

    @Override
    public Boolean apply(Number input) {
        return Double.isNaN(input.doubleValue());
    }

    @Override
    public Boolean apply(IComplexNumber input) {
        return Double.isNaN(input.absoluteValue().doubleValue());
    }
}
