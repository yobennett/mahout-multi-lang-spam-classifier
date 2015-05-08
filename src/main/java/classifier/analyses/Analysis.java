package classifier.analyses;

import com.google.common.collect.Multiset;
import org.apache.mahout.math.Vector;

import java.io.IOException;

public interface Analysis {

    Multiset<String> words(String text) throws IOException;

    Vector instanceVector();

}
