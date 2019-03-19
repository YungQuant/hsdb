#include "data.h"

#include <iostream>
#include <string>
#include <sstream>

#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include "/usr/include/boost/variant.hpp"
#include <boost/property_tree/ptree.hpp>
#include <boost/algorithm/string.hpp>


data::data(int store_len)
{
    len = store_len;
    sync = false;
}

auto fetch_value = [](std::map<std::string, std::map<std::string, std::vector<std::string>>> x, std::string symbol, std::string tag, std::string size){
    std::vector<std::string> y = x[symbol][tag];
    std::vector<std::string> z = {y[0], size};
    return z;
};


boost::property_tree::ptree data::json(std::string msg){
    boost::property_tree::ptree pt;
    std::stringstream ss;
    ss << msg;
    boost::property_tree::read_json(ss, pt);
    return pt;
}

int data::cut_book(boost::property_tree::ptree const & data){
    boost::property_tree::ptree::const_iterator end = data.end();
    std::string symbol, side, ordID;
    bool ok_update = false;

    for(boost::property_tree::ptree::const_iterator it = data.begin(); it != end; ++it){
        if(it->first == "symbol"){
            symbol = (*it).second.get_value<std::string>();
        }
        if(it->first == "id"){
            ordID = (*it).second.get_value<std::string>();
        }
        if(it->first == "side"){
            side = (*it).second.get_value<std::string>();
            ok_update = true;
        }

        if(ok_update == true){
            if(side == "Buy"){
                bids[symbol].erase(ordID);
            } else {
                asks[symbol].erase(ordID);
            }
        }

        cut_book(it->second);
    }

    return 0;
}


int data::refresh_book(boost::property_tree::ptree const & data){
    boost::property_tree::ptree::const_iterator end = data.end();
    std::vector<std::string> temp;
    std::string symbol, side, ordID, size;
    bool ok_update = false;

    for(boost::property_tree::ptree::const_iterator it = data.begin(); it != end; ++it){
        std::cout << "HUUUU: " << it->first << "\tHEABITCH: " << it->second.get_value<std::string>() << std::endl;
        if(it->first == "symbol"){
            symbol = (*it).second.get_value<std::string>();
        }
        if(it->first == "id"){
            ordID = (*it).second.get_value<std::string>();
        }
        if(it->first == "side"){
            side = (*it).second.get_value<std::string>();
        }
        if(it->first == "size"){
            size = (*it).second.get_value<std::string>();
            ok_update = true;
        }

        if(ok_update == true){
            if(side == "Buy"){
                bids[symbol][ordID][1] = size;
            } else {
                asks[symbol][ordID][1] = size;
            }

            ok_update = false;
        }
        try{
            refresh_book(it->second);
        } catch (const std::exception& e){

        }
    }
    return 0;
}

int data::put_book(boost::property_tree::ptree const & data) {
    boost::property_tree::ptree::const_iterator end = data.end();
    
    std::string symbol, side, ordID, price, size;
    bool ok_deposit = false;

    for(boost::property_tree::ptree::const_iterator it = data.begin(); it != end; ++it){
        if(it->first == "symbol"){
            symbol = (*it).second.get_value<std::string>();
        }
        if(it->first == "id"){
            ordID = (*it).second.get_value<std::string>();
        }
        if(it->first == "side"){
            side = (*it).second.get_value<std::string>();
        }
        if(it->first == "size"){
            size = (*it).second.get_value<std::string>();
        }
        if(it->first == "price"){
            price = (*it).second.get_value<std::string>();
            ok_deposit = true;
        }
        
        if(ok_deposit == true){
            if(side == "Buy"){
                bids[symbol][ordID].push_back(price);
                bids[symbol][ordID].push_back(size);
            } else {
                asks[symbol][ordID].push_back(price);
                asks[symbol][ordID].push_back(size);
            }

            ok_deposit = false;
        }

        put_book(it->second);
    }

    return 0;
}

int data::twist_book(boost::property_tree::ptree const & data) {
    boost::property_tree::ptree::const_iterator end = data.end();
    
    std::string symbol, side, ordID, price, size;
    bool ok_deposit = false;

    for(boost::property_tree::ptree::const_iterator it = data.begin(); it != end; ++it){
        
        if(it->first == "symbol"){
            symbol = (*it).second.get_value<std::string>();
        }
        if(it->first == "id"){
            ordID = (*it).second.get_value<std::string>();
        }
        if(it->first == "side"){
            side = (*it).second.get_value<std::string>();
        }
        if(it->first == "size"){
            size = (*it).second.get_value<std::string>();
        }
        if(it->first == "price"){
            price = (*it).second.get_value<std::string>();
            ok_deposit = true;
        }
        
        if(ok_deposit == true){
            if(side == "Buy"){
                bids[symbol][ordID].push_back(price);
                bids[symbol][ordID].push_back(size);
            } else {
                asks[symbol][ordID].push_back(price);
                asks[symbol][ordID].push_back(size);
            }

            ok_deposit = false;
        }

        twist_book(it->second);
    }
    return 0;
}

int data::cyclone(boost::property_tree::ptree const & pt)
{
    boost::property_tree::ptree::const_iterator end = pt.end();
     
    bool partial_ = false, update_ = false, insert_ = false, delete_ = false;
    bool obook = false;
    
    for(boost::property_tree::ptree::const_iterator it = pt.begin(); it != end; ++it){
        if(partial_ == true){
            if(it->first == "data"){
                twist_book(it->second);
                partial_ = false;
                obook = false;
            }
        }

        if(insert_ == true){
            if(it->first == "data"){
                put_book(it->second);
                insert_ = false;
                obook = false;
            }
        }


        if(update_ == true){
            if(it->first == "data"){
                refresh_book(it->second);
                update_ = false;
                obook = false;
            }
        }

        if(delete_ == true){
            if(it->first == "data"){
                cut_book(it->second);
                delete_ = false;
                obook = false;
            }
        }

        if(obook == true){
            if(it->first == "action"){
                if(it->second.get_value<std::string>() == "partial"){
                    partial_ = true;
                }
                if(it->second.get_value<std::string>() == "update"){
                    //update_ = true;    
                }
                if(it->second.get_value<std::string>() == "insert"){
                    insert_ = true;
                }
                if(it->second.get_value<std::string>() == "delete"){
                    delete_ = true;
                }
            }
        }

        if(it->first == "table"){
            if(it->second.get_value<std::string>() == "orderBookL2"){
                obook = true;
            }
        }

        if(bids["XBTUSD"].size() > 0 && asks["XBTUSD"].size() > 0){
            sync = true;
        }

        cyclone(it->second);
    }
    return 0;
}

int data::book_fetch(std::string symbol, int depth)
{
    
    return 0;
}