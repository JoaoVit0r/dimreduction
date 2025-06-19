package agn;

import java.util.List;
import java.util.Vector;
import java.util.concurrent.atomic.AtomicInteger;

public class ThreadManager {
    public interface TargetProcessor {
        void process(int targetIndex);
    }

    /**
     * By Slice Groups: Split targets into N contiguous groups (N = number of threads),
     * each thread processes its group sequentially. All threads start at once.
     * This is static chunking.
     */
    public void executeThreadsBySliceGroups(List<Integer> targets, TargetProcessor processor, int numberOfThreads) {
        int total = targets.size();
        int chunk = (total + numberOfThreads - 1) / numberOfThreads;
        Vector<Thread> threads = new Vector<>();
        for (int t = 0; t < numberOfThreads; t++) {
            final int start = t * chunk;
            final int end = Math.min(start + chunk, total);
            Thread thread = new Thread(() -> {
                for (int i = start; i < end; i++) {
                    processor.process(targets.get(i));
                }
            });
            threads.add(thread);
            thread.start();
        }
        for (Thread thread : threads) {
            try {
                thread.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * By Stride Groups: Each thread gets every N-th target (N = number of threads).
     * All threads start at once, each processes its own strided list sequentially.
     */
    public void executeThreadsByStrideGroups(List<Integer> targets, TargetProcessor processor, int numberOfThreads) {
        int total = targets.size();
        Vector<Thread> threads = new Vector<>();
        for (int t = 0; t < numberOfThreads; t++) {
            final int threadIndex = t;
            Thread thread = new Thread(() -> {
                for (int i = threadIndex; i < total; i += numberOfThreads) {
                    processor.process(targets.get(i));
                }
            });
            threads.add(thread);
            thread.start();
        }
        for (Thread thread : threads) {
            try {
                thread.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * By Dynamic Assignment: All threads share a synchronized index, each grabs the next available target.
     * This is dynamic load balancing (work-stealing).
     */
    public void executeThreadsByDynamicAssignment(List<Integer> targets, TargetProcessor processor, int numberOfThreads) {
        int total = targets.size();
        AtomicInteger nextIndex = new AtomicInteger(0);
        Vector<Thread> threads = new Vector<>();
        for (int t = 0; t < numberOfThreads; t++) {
            Thread thread = new Thread(() -> {
                int i;
                while ((i = nextIndex.getAndIncrement()) < total) {
                    processor.process(targets.get(i));
                }
            });
            threads.add(thread);
            thread.start();
        }
        for (Thread thread : threads) {
            try {
                thread.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}
