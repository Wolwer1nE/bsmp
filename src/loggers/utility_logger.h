// Utility logger. Replaces any control flow outputs
// writes to stderr
#ifndef UTILITY_LOGGER_H
#define UTILITY_LOGGER_H

#include <cstdarg>
#include <cstdio>
#include <string>

#ifndef ACTIVE_UTILITY_LOG_LEVEL
#define ACTIVE_UTILITY_LOG_LEVEL UtilityLogLevel::INFO
#endif

enum class UtilityLogLevel {
    TRACE = 0,
    DEBUG,
    INFO,
    WARN,
    ERROR,
    FATAL
};

class UtilityLogger {
   private:
    UtilityLogger& instance() {
        static UtilityLogger logger;
        return logger;
    }
    UtilityLogger() = default;

   public:
    UtilityLogger(const UtilityLogger&) = delete;
    UtilityLogger& operator=(const UtilityLogger&) = delete;
    static void log_output(int loglevel, const char* file, int line, const char* fmt, ...) {
        if (!fmt) {
            return;
        }
        constexpr static const char* levelstring[6] = {
            "TRACE", "DEBUG", "INFO",
            "WARN", "ERROR", "FATAL"};
        fprintf(stderr, "[%s] %s:%d: ", levelstring[loglevel], file, line);
        va_list args;
        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        va_end(args);
        fprintf(stderr, "\n");
    }
};

#define LOG(level, fmt, ...)                                                                            \
    do {                                                                                                \
        if constexpr (static_cast<int>(level) >= static_cast<int>(ACTIVE_UTILITY_LOG_LEVEL)) {          \
            UtilityLogger::log_output(static_cast<int>(level), __FILE__, __LINE__, fmt, ##__VA_ARGS__); \
            if constexpr (static_cast<int>(level) <= static_cast<int>(UtilityLogLevel::INFO)) {         \
                fflush(stderr);                                                                         \
            }                                                                                           \
        }                                                                                               \
    } while (0)

#define LOG_TRACE(fmt, ...) LOG(UtilityLogLevel::TRACE, fmt, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) LOG(UtilityLogLevel::DEBUG, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) LOG(UtilityLogLevel::INFO, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...) LOG(UtilityLogLevel::WARN, fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) LOG(UtilityLogLevel::ERROR, fmt, ##__VA_ARGS__)
#define LOG_FATAL(fmt, ...) LOG(UtilityLogLevel::FATAL, fmt, ##__VA_ARGS__)

#endif  // UTILITY_LOGGER_H