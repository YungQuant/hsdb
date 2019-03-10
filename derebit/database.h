#ifndef DATABASE_H
#define DATABASE_H

#include <future>
#include <unordered_map>

#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include "/usr/include/boost/variant.hpp"
#include <boost/property_tree/ptree.hpp>
#include <boost/algorithm/string.hpp>



class database{
	
	private:
		std::string old_msg;
	
		
	public:
		database();
	
		std::future<void> push_msg(std::string message);
		
		std::unordered_map<std::string, std::vector<std::vector<double>>> raw_data;
	
		int deposit(boost::property_tree::ptree const& pt, std::vector<std::vector<double>> &book);
		bool check_msg(std::string message);
	
		int write_book(std::string key, std::vector<std::vector<double>> value);
		std::vector<std::vector<double>> get_bids(std::string key);
	
		std::vector<std::string> get_instruments();
	
};


#endif