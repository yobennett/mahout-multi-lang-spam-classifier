package classifier;

import java.io.IOException;

public interface Classifier<T, V> {
	
	V classify(T input) throws IOException;
	
}
