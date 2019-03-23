#ifndef DATA_H
#define DATA_H

#include <string>
#include <vector>
#include <map>

#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include "/usr/include/boost/variant.hpp"
#include <boost/property_tree/ptree.hpp>
#include <boost/algorithm/string.hpp>



class data {

    private:
        int len;
        
    public:
        data(int store_len);
        std::map<std::string, std::map<std::string, std::vector<std::string>>> bids;
        std::map<std::string, std::map<std::string, std::vector<std::string>>> asks;
        std::map<std::string, std::vector<std::vector<double>>> current_book;

        std::map<std::string, std::vector<std::string>> OBOOK;

        bool sync;

        boost::property_tree::ptree json(std::string message);
        int cyclone(boost::property_tree::ptree const & pt);
        int twist_book(boost::property_tree::ptree const & pt);
        int refresh_book(boost::property_tree::ptree const & pt);
        int cut_book(boost::property_tree::ptree const & pt);
        int put_book(boost::property_tree::ptree const & pt);

        int book_fetch(std::string symbol, int depth);

        int book(std::string symbol);

};


#endif
