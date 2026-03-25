// Same to Utility logger. The reason for its existence is to separate program
// control flow logs from metrics collected for analysis 

/* FIXME: there could've been a generic Logger class with different sinks. 
   I didn't bother, maybe you will */

#ifndef PERFORMANCE_LOGGER_H
#define PERFORMANCE_LOGGER_H

#include <fstream>
#include <string>
#include <cstdarg>

#define ACTIVE_METRICS_LOG_LEVEL MetricsLogLevel::PERF

enum class MetricsLogLevel {
    CONV = 0, // Metrics for convergence analysis, impact performance
    PERF,
    NONE
};

class MetricsLogger {
private:
    static MetricsLogger& instance() {
        static MetricsLogger logger;
        return logger;
    }
    std::ofstream ofs;
    MetricsLogger() = default;
public:
    MetricsLogger(const MetricsLogger&) = delete;
    MetricsLogger& operator=(const MetricsLogger&) = delete;
    // unlike UtilityLogger, this has to be initialized because of a file usage
    static void init_logger(const std::string& filepath) {
        if (instance().ofs.is_open()) instance().ofs.close();
        instance().ofs.open(filepath, std::ios_base::app);
    }
    static void log_output(const char* fmt, ...) {
        if (!fmt) {
            return;
        }
        
        va_list args;
        va_start(args, fmt);
        std::string buf(256, 0);
        while (true) {
            int size = vsnprintf(&buf[0], buf.size(), fmt, args);
            size_t new_sz = (size < 0) ? (buf.size() * 2) : size;
            if (new_sz >= buf.size()) {
                buf.resize(new_sz + 1);
            }
            else {
                buf.resize(new_sz);
                break;
            }
        }
        va_end(args);
        instance().ofs << buf << '\n';
    }
};

#define LOG_METRIC(level, fmt, ...) do { \
    if constexpr (static_cast<int>(level) >= static_cast<int>(ACTIVE_METRICS_LOG_LEVEL)) { \
        MetricsLogger::log_output(fmt, ##__VA_ARGS__); \
    }} while(0)


#define LOG_CONV(fmt, ...) LOG_METRIC(MetricsLogLevel::CONV, fmt, ##__VA_ARGS__)
#define LOG_PERF(fmt, ...) LOG_METRIC(MetricsLogLevel::PERF, fmt, ##__VA_ARGS__)
#endif // PERFORMANCE_LOGGER_H