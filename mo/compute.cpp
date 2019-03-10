#include "compute.h"
#include <string>
#include <sstream>

#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include "/usr/include/boost/variant.hpp"
#include <boost/property_tree/ptree.hpp>
#include <boost/algorithm/string.hpp>

#include <iostream>

Compute::Compute(std::vector<std::string> data)
{
		
}



boost::property_tree::ptree parser(std::string message)
{
	boost::property_tree::ptree hold;
	
	std::stringstream ss;
	ss << message;
	
	boost::property_tree::read_json(ss, hold);
	
	return hold;
}







std::string Compute::Call(std::string message)
{
	std::string result;
	
	boost::property_tree::ptree data = parser(message);
	
	std::string table_type = data.get<std::string>("table");
	std::string action = data.get<std::string>("action");
	
	std::cout << table_type << "\t" << action << std::endl;
	
	return result;
}