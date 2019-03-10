#include "database.h"

#include <iostream>
#include <sstream>
#include <future>
#include <vector>
#include <algorithm>

#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include "/usr/include/boost/variant.hpp"
#include <boost/property_tree/ptree.hpp>
#include <boost/algorithm/string.hpp>

using namespace boost::property_tree;

database::database()
{

}

bool check_bcol(std::string key)
{
    bool res = false;
    if(key == "quantity" || key == "amount" || key == "price" || key == "cm" || key == "cm_amount")
    {
        res = true;   
    }
    return res;
}

std::vector<std::string> database::get_instruments()
{
    std::vector<std::string> result;
    result.reserve(raw_data.size());
    for(auto kv : raw_data)
    {
        result.push_back(kv.first);   
    }
    return result;
}

int database::write_book(std::string key, std::vector<std::vector<double>> value)
{
    std::cout << " Depositing book for " << key << " of size: " << value.size() << std::endl;
    raw_data[key] = value;
    return 0;
}

std::vector<std::vector<double>> database::get_bids(std::string key)
{
    std::vector<std::vector<double>> res = raw_data[key];
    std::cout << "Total Book Size: " << res.size() << std::endl;
    return res;
}


bool database::check_msg(std::string message)
{
    bool res = false;
    if(message != old_msg)
    {
        old_msg = message;
        res = true;
    }
    return res;
}

int database::deposit(ptree const& pt, std::vector<std::vector<double>> &book) 
{
    std::vector<double> temp;
    
    ptree::const_iterator end = pt.end();
    for (ptree::const_iterator it = pt.begin(); it != end; ++it) {
        //std::cout << it->first << " : " << it->second.get_value<std::string>() << " || ";
        
        if(check_bcol(it->first) == true)
        {
            temp.push_back(atof(it->second.get_value<std::string>().c_str()));   
        }
        
        if(it->first == "cm_amount")
        {
            book.push_back(temp);
            temp.clear();
        }
        deposit(it->second, book);
    }
            
    return 0; 
}

void cyclone(database * self, ptree const& pt, std::string instrument, std::vector<std::vector<double>> book)
{
    std::string switching;
    ptree::const_iterator end = pt.end();
    for (ptree::const_iterator it = pt.begin(); it != end; ++it) {
        if(it->first == "instrument")
        {
            instrument = it->second.get_value<std::string>().c_str();
        }
        if(it->first == "bids")
        {   
            (*self).deposit(it->second, book);
            (*self).write_book(instrument, book);
            book.clear();
            (*self).get_bids(instrument);
        }
        
        cyclone(self, it->second, instrument, book);
      
    }
}


std::future<void> database::push_msg(std::string message)
{
	ptree pt;
	std::stringstream ss;
	ss << message;
	read_json(ss, pt);
    
    std::string instrument;
	std::vector<std::vector<double>> book;
    
    std::future<void> hold(std::async(cyclone, this, pt, instrument, book));
    
    
	return hold;
}
