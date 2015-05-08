package classifier.analyses;

import org.apache.mahout.math.Vector;

public interface Analysis<T> {

    Vector instanceVector();

}
