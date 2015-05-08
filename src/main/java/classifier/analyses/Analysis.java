package classifier.analyses;

import com.google.common.collect.Multiset;
import org.apache.mahout.math.Vector;

import java.io.IOException;

public interface Analysis<T> {

    Multiset<T> words(T text) throws IOException;

    Vector instanceVector();

}
