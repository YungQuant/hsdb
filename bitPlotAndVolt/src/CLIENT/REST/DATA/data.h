#ifndef DATA_H
#define DATA_H

#include <string>
#include <vector>
#include <map>
#include <fstream>

#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include "/usr/include/boost/variant.hpp"
#include <boost/property_tree/ptree.hpp>
#include <boost/algorithm/string.hpp>



class data {

    private:
        int len;
        double oldBid;
        double oldAsk;
        std::map<double, double> vol_bids;
        std::map<double, double> vol_asks;
		double timeStart;
		double currTimeNow;

    public:
        data(int store_len, int dt);
        std::map<std::string, std::map<std::string, std::vector<std::string>>> bids;
        std::map<std::string, std::map<std::string, std::vector<std::string>>> asks;

        std::map<std::string, std::vector<std::vector<double>>> current_book;

        std::map<std::string, std::vector<std::string>> OBOOK;
        std::map<std::string, std::vector<std::vector<double>>> XY;

        std::vector<std::vector<double>> Z;

        int time_len;
        bool sync;
        bool plot_sync;
        int quant_sync;
        std::vector<double> lemp, hemp, temp;
		std::vector<double> bidAverage;
		std::vector<double> askAverage;    
		int bid_vol_len;
		int ask_vol_len;    

		double now();

        boost::property_tree::ptree json(std::string message);
        int cyclone(boost::property_tree::ptree const & pt);
        int twist_book(boost::property_tree::ptree const & pt);
        int refresh_book(boost::property_tree::ptree const & pt);
        int cut_book(boost::property_tree::ptree const & pt);
        int put_book(boost::property_tree::ptree const & pt);

        int book(std::string symbol);
        int write_mesh(std::string symbol);
        int clean_table(int limit);
        int fix_price_axis();

        int prep_book(std::ofstream & writer, std::string symbol);

        int __call__(std::string symbol);
        int __volt__(std::string tag, double price);

};


#endif
