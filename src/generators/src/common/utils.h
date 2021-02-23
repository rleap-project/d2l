
#pragma once

#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <boost/functional/hash.hpp>

namespace sltp::utils {

template<typename Container>
struct container_hash {
    std::size_t operator()(const Container& c) const {
        return boost::hash_range(c.begin(), c.end());
    }
};

std::ofstream get_ofstream(const std::string &filename);
std::ifstream get_ifstream(const std::string &filename);

float read_time_in_seconds(bool add_stime = true);

inline std::string normal() { return "\x1B[0m"; }
inline std::string red() { return "\x1B[31;1m"; }
inline std::string green() { return "\x1B[32;1m"; }
inline std::string yellow() { return "\x1B[33;1m"; }
inline std::string blue() { return "\x1B[34;1m"; }
inline std::string magenta() { return "\x1B[35;1m"; }
inline std::string cyan() { return "\x1B[36;1m"; }
inline std::string error() { return "\x1B[31;1merror: \x1B[0m"; }
inline std::string warning() { return "\x1B[35;1mwarning: \x1B[0m"; }
inline std::string internal_error() { return "\x1B[31;1minternal error: \x1B[0m"; }

inline std::string cmdline(int argc, const char **argv) {
    std::string cmd(*argv);
    for( int j = 1; j < argc; ++j )
        cmd += std::string(" ") + argv[j];
    return cmd;
}

template<typename T> inline std::string as_string(const T &container) {
    bool need_comma = false;
    std::string str("{");
    for( typename T::const_iterator it = container.begin(); it != container.end(); ++it ) {
        if( need_comma ) str += ",";
        str += std::to_string(*it);
        need_comma = true;
    }
    return str + "}";
}

inline std::string as_string(const std::vector<bool> &bitmap) {
    bool need_comma = false;
    std::string str("{");
    for( int i = 0; i < int(bitmap.size()); ++i ) {
        if( bitmap[i] ) {
            if( need_comma ) str += ",";
            str += std::to_string(i);
            need_comma = true;
        }
    }
    return str + "}";
}

inline std::string as_string(const int *block) {
    bool need_comma = false;
    std::string str("{");
    for( int i = 0, block_size = *block; i < block_size; ++i ) {
        if( need_comma ) str += ",";
        str += std::to_string(block[1 + i]);
        need_comma = true;
    }
    return str + "}";
}

template<typename T>
std::vector<T> split(const std::string& line) {
    std::istringstream is(line);
    return std::vector<T>(std::istream_iterator<T>(is), std::istream_iterator<T>());
}


}
