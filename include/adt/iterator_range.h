#pragma once

namespace tfbe
{
// TODO: implement it
template <typename IteratorT, typename MT>
class mapped_iterator
{
public:
    mapped_iterator(IteratorT iter) : iter_(iter) {}
    MT operator*()
    {
        return MT(*iter_);
    }

    bool operator==(const mapped_iterator<IteratorT, MT>& other)
    {
        return iter_ == other.iter_;
    }
    bool operator!=(const mapped_iterator<IteratorT, MT>& other)
    {
        return iter_ != other.iter_;
    }

    mapped_iterator operator++()
    {
        return mapped_iterator(++iter_);
    }

    mapped_iterator operator++(int)
    {
        return mapped_iterator(iter_++);
    }

private:
    // TODO:
    // map function
    IteratorT iter_;
};

template <typename IterT>
class iterator_range
{
public:
    iterator_range(IterT begin, IterT end) : begin_(begin), end_(end) {}

    IterT begin()
    {
        return begin_;
    }
    IterT end()
    {
        return end_;
    }

private:
    IterT begin_;
    IterT end_;
};

template <typename IterT>
iterator_range<IterT> make_range(IterT begin, IterT end)
{
    return iterator_range<IterT>(begin, end);
}

template <typename ContainerT>
iterator_range<typename ContainerT::const_iterator> make_range(const ContainerT& c)
{
    return iterator_range<typename ContainerT::const_iterator>(std::begin(c), std::end(c));
}

template <typename MT, typename ContainerT>
iterator_range<mapped_iterator<typename ContainerT::const_iterator, MT>> make_map_range(const ContainerT& c)
{
    return iterator_range<mapped_iterator<typename ContainerT::const_iterator, MT>>(
        mapped_iterator<typename ContainerT::const_iterator, MT>(std::begin(c)),
        mapped_iterator<typename ContainerT::const_iterator, MT>(std::end(c)));
}

} // namespace tfbe
