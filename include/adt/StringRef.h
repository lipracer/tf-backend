#pragma once

#include <iosfwd>
#include <string>

#include <stddef.h>
#include <string.h>

namespace tfbe
{
class StringRef
{
public:
    using CharT = char;
    using iterator = const CharT*;
    using const_iterator = const CharT*;

    StringRef() : size_(0), str_(nullptr) {}

    StringRef(const char* str) : size_(strlen(str)), str_(str) {}

    template <size_t N>
    StringRef(const char (&str)[N]) : size_(N), str_(str)
    {
    }

    StringRef(const std::string& str) : size_(str.size()), str_(str.c_str()) {}

    const CharT* c_str() const
    {
        return str_;
    }

    bool empty() const
    {
        return !size_ || !str_;
    }

    size_t size() const
    {
        return size_;
    }

    std::string str() const
    {
        return std::string(str_, str_ + size_);
    }

    bool operator==(const char* str) const
    {
        StringRef other(str);
        return *this == other;
    }

    bool operator==(StringRef other) const
    {
        return size_ == other.size_ && (str_ == other.str_ || !strcmp(str_, other.str_));
    }

    bool operator!=(StringRef other) const
    {
        return !(*this == other);
    }

private:
    size_t size_;
    const CharT* str_;
};

inline std::ostream& operator<<(std::ostream& os, StringRef str)
{
    os << str.str();
    return os;
}

} // namespace tfbe
