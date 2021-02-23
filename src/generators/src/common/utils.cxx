

#include <common/utils.h>

#include <sys/resource.h>

namespace sltp::utils {

std::ofstream get_ofstream(const std::string &filename) {
    std::ofstream stream(filename.c_str());
    if(stream.fail()) {
        throw std::runtime_error(sltp::utils::error() + "opening file '" + filename + "'");
    }
    return stream;
}

std::ifstream get_ifstream(const std::string &filename) {
    std::ifstream stream(filename.c_str());
    if(stream.fail()) {
        throw std::runtime_error(sltp::utils::error() + "opening file '" + filename + "'");
    }
    return stream;
}

float read_time_in_seconds(bool add_stime) {
    struct rusage r_usage;
    float time = 0;

    getrusage(RUSAGE_SELF, &r_usage);
    time += float(r_usage.ru_utime.tv_sec) +
            float(r_usage.ru_utime.tv_usec) / float(1e6);
    if (add_stime) {
        time += float(r_usage.ru_stime.tv_sec) +
                float(r_usage.ru_stime.tv_usec) / float(1e6);
    }

    getrusage(RUSAGE_CHILDREN, &r_usage);
    time += float(r_usage.ru_utime.tv_sec) +
            float(r_usage.ru_utime.tv_usec) / float(1e6);
    if (add_stime) {
        time += float(r_usage.ru_stime.tv_sec) +
                float(r_usage.ru_stime.tv_usec) / float(1e6);
    }

    return time;
}

} // namespaces