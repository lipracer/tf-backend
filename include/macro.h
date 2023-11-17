#pragma once

#define BE_EXPORT __attribute__((__visibility__("default")))

#define be_unreachable(...)             \
    do                                  \
    {                                   \
        int a = 0;                      \
        *reinterpret_cast<int*>(a) = 0; \
    } while (0);

#define EXPECT(p, ...)        \
    do                        \
    {                         \
        if (!(p))             \
            be_unreachable(); \
    } while (0)

#define EXPECT_EQ(l, r, ...)  \
    do                        \
    {                         \
        if ((l) != (r))       \
            be_unreachable(); \
    } while (0)

#define EXPECT_NQ(l, r, ...)  \
    do                        \
    {                         \
        if ((l) == (r))       \
            be_unreachable(); \
    } while (0)
