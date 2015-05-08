package classifier;

import java.io.IOException;

public interface Classifier<T, V> {
	
	T classify(V input) throws IOException;
	
}
