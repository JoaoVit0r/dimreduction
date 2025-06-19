package utilities;

import java.time.Instant;
import java.util.HashMap;

public class Timer {
    private HashMap<String, Long> timers = new HashMap<>();

    /**
     * Starts a timer with the given name.
     * If the timer is already running, it logs a message.
     *
     * @param name The name of the timer to start.
     * @deprecated This method is deprecated, no longer used to avoid performance overhead.
     */
    public void start(String name) {
        if (true) {
            return;
        }
        // if (timers.containsKey(name)) {
        //     IOFile.PrintlnAndLog("Timer " + name + " is already running.", "timing/timers.log");
        // } else {
        //     timers.put(name, System.currentTimeMillis());
        //     IOFile.PrintlnAndLog("Timer " + name + " started at " + Instant.now(), "timing/timers.log");
        // }
    }

    /**
     * Ends a timer with the given name and logs the duration.
     * If the timer was not started, it logs a message.
     *
     * @param name The name of the timer to end.
     * @deprecated This method is deprecated, no longer used to avoid performance overhead.
     */
    public void end(String name) {
        if (true) {
            return;
        }
        // if (!timers.containsKey(name)) {
        //     IOFile.PrintlnAndLog("Timer " + name + " was not started.", "timing/timers.log");
        // } else {
        //     long startTime = timers.remove(name);
        //     long endTime = System.currentTimeMillis();
        //     double duration = (endTime - startTime) / 1000.0;
        //     IOFile.PrintlnAndLog("Timer " + name + " ended at " + Instant.now(), "timing/timers.log");
        //     IOFile.PrintlnAndLog("Duration for " + name + ": " + String.format("%.4f", duration) + " seconds", "timing/timers.log");
        // }
    }
}
