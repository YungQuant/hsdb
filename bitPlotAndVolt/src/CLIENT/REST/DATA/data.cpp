#include "./data.h"

#include <iostream>
#include <future>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <math.h>
#include <time.h>

#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include "/usr/include/boost/variant.hpp"
#include <boost/property_tree/ptree.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/range/combine.hpp>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/range.hpp>


data::data(int store_len, int dt)
{
    len = store_len - 1;
    time_len = dt;
    sync = false;
    quant_sync = false;
    plot_sync = false;
    oldBid = 0;
    oldAsk = 0;
	timeStart = now();
	currTimeNow = now();
}

auto cut_vector = [](std::vector<std::vector<double>> x, int c){
    std::vector<std::vector<double>> y;
    int start = 0;
    for(auto & t : x){
        if(start > c){
            y.push_back(t);
        }
        start = 1;
    }
    return y;
};

auto NPMESHGRID = [](std::vector<std::vector<double>> x){
    int m = x.size();
    int n = x[0].size();
    std::map<std::string, std::vector<std::vector<double>>> z;
    std::vector<double> tempW, tempX;
    for(int i = 0; i < m; ++i){
        for(int j = 0; j < n; ++j){
            tempW.push_back(i+1);
            tempX.push_back(j+1);
        }
        z["X"].push_back(tempW);
        z["Y"].push_back(tempX);
        tempW.clear();
        tempX.clear();
    }
    return z;
};


boost::property_tree::ptree data::json(std::string msg){
    boost::property_tree::ptree pt;
    std::stringstream ss;
    ss << msg;
    boost::property_tree::read_json(ss, pt);
    return pt;
}

double data::now(){
    time_t time1;
    time(&time1);
    return (double) time1;
};

int data::__call__(std::string symbol)
{

    if(Z.size() < time_len + 20){
        write_mesh(symbol);
        XY = NPMESHGRID(Z);
        plot_sync = true;
    } else {
        plot_sync = false;
        clean_table(20);
    }


    return 0;
}


double calc_vol(std::map<double, double> x, double n)
{
	double t_sum = 0, tt = 0;
	int ct = 0;
	for(auto kv = x.rbegin(); kv != x.rend(); ++kv){
		if(ct == 0){
			tt = kv->first;
			t_sum += kv->second;
		}
		if(tt - kv->first <= n && ct > 0){
			t_sum += kv->second;
		} else {
			break;
		}
		ct += 1;
	}
	return pow(t_sum / (ct - 1), 0.5);
}

int data::__volt__(std::string tag, double price){
    //std::cout << price << std::endl;
	std::vector<double> big_three;    
	std::vector<std::future<double>> res;
	int epoch = 0;
    double start_time = 0;

    if(tag == "bid"){
        if(price != oldBid){
			if(oldBid != 0){            	
				vol_bids[now()] = (price/oldBid) - 1;
				res.push_back(std::async(calc_vol, vol_bids, 60));
				res.push_back(std::async(calc_vol, vol_bids, 600));
				res.push_back(std::async(calc_vol, vol_bids, 3600));
				bidAverage.clear();
				for(auto & tasks : res){
					bidAverage.push_back(tasks.get());				
				}
				if(currTimeNow - timeStart > 5000){
					big_three.clear();
					for(auto kv : vol_bids){
						if(currTimeNow - kv.first >= 3600) {
							big_three.push_back(kv.first);
						} else {
							break;
						}
					}
					for(auto & uu : big_three){
						vol_bids.erase(uu);
					}
				}
			}
            oldBid = price;
        }
    } else {
        if(price != oldAsk){
			if(oldAsk != 0){            	
				vol_asks[now()] = (price/oldAsk) - 1;
				res.push_back(std::async(calc_vol, vol_asks, 60));
				res.push_back(std::async(calc_vol, vol_asks, 600));
				res.push_back(std::async(calc_vol, vol_asks, 3600));
				askAverage.clear();
				for(auto & tasks : res){
					askAverage.push_back(tasks.get());
				}
				if(currTimeNow - timeStart > 5000){
					big_three.clear();
					for(auto kv : vol_asks){
						if(currTimeNow - kv.first >= 3600) {
							big_three.push_back(kv.first);
						} else {
							break;
						}
					}
					for(auto & uu : big_three){
						vol_asks.erase(uu);
					}
				}
			}
            oldAsk = price;
        }
    }
	bid_vol_len = vol_bids.size();
	ask_vol_len = vol_asks.size();
	currTimeNow = now();
    return 0;
}

int data::fix_price_axis()
{
    int n = 9;
    double a = lemp[0];
    double b = lemp[lemp.size() - 1];
    double dx = (b - a) / (n - 1);
    dx = floor(dx * 100) / 100;
    //std::cout << "DX: " << dx << std::endl;
    lemp.clear();
    for(int i = 0; i < n; ++i){
        lemp.push_back(a + i * dx);
    }
    return 0;
}

int data::clean_table(int length){
    for(int i = 0; i < length; ++i){
        XY["X"].erase(XY["X"].begin()+i);
        XY["Y"].erase(XY["Y"].begin()+i);
        Z.erase(Z.begin()+i);
    }
    plot_sync = true;
    return 0;
}


int data::write_mesh(std::string symbol)
{
    lemp.clear();
    //hemp.clear();
    temp.clear();
    std::vector<std::string> x;
    double bid = 0, ask = 0;
    int ct = 0;
    int dt = 0;

    for(auto kv : OBOOK){
        OBOOK[kv.first].clear();
    }
    for(auto kv : bids[symbol]){
        //hemp.push_back(dt);
        bid += atof(bids[symbol][kv.first][0].c_str()) * atof(bids[symbol][kv.first][1].c_str());
        lemp.push_back(atof(bids[symbol][kv.first][0].c_str()));
        temp.push_back(bid);
        if(ct >= len){
            break;
        }
        ct += 1;
        dt += 1;
    }
    __volt__("bid", lemp[0]);
    std::reverse(lemp.begin(), lemp.end());
    std::reverse(temp.begin(), temp.end());
    ct = 0;
    for(auto kv = asks[symbol].rbegin(); kv != asks[symbol].rend(); ++kv){
        x = kv->second;
        //hemp.push_back(dt);
        if(ask == 0){
            __volt__("ask", atof(x[0].c_str()));
        }
        ask += atof(x[0].c_str()) * atof(x[1].c_str());
        lemp.push_back(atof(x[0].c_str()));
        temp.push_back(ask);
        if(ct >= len){
            break;
        }
        ct += 1;
        dt += 1;
    }
    Z.push_back(temp);
    fix_price_axis();

    return 0;
}

int data::prep_book(std::ofstream & writer, std::string symbol)
{
    std::string prices = "", volumes = "";
    for(auto kv : bids[symbol]){
        prices += bids[symbol][kv.first][0] + ",";
        volumes += bids[symbol][kv.first][1] + ",";
    }
    prices.pop_back();
    volumes.pop_back();

    writer << prices << "\n" << volumes << "\n";

    prices = "", volumes = "";
    for(auto kv : asks[symbol]){
        prices += asks[symbol][kv.first][0] + ",";
        volumes += asks[symbol][kv.first][1] + ",";
    }
    prices.pop_back();
    volumes.pop_back();

    writer << prices << "\n" << volumes << "\n";

    return 0;
}

int data::book(std::string symbol)
{
    for(auto kv : OBOOK){
        OBOOK[kv.first].clear();
    }
    std::string qid;
    int ct = 0;
    for(auto kv : bids[symbol]){
        OBOOK["BidPrice"].push_back(bids[symbol][kv.first][0]);
        OBOOK["BidVol"].push_back(bids[symbol][kv.first][1]);
        if(ct >= len){
            break;
        }
        ct += 1;
    }
    ct = 0;
    std::vector<std::string> x;
    for(auto kv = asks[symbol].rbegin(); kv != asks[symbol].rend(); ++kv){
        x = kv->second;
        OBOOK["AskPrice"].push_back(x[0]);
        OBOOK["AskVol"].push_back(x[1]);
        if(ct >= len){
            break;
        }
        ct += 1;
    }
    //std::reverse(OBOOK["BidPrice"].begin(), OBOOK["BidPrice"].end());
    //std::reverse(OBOOK["BidVol"].begin(), OBOOK["BidVol"].end());
    return 0;
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
    quant_sync = false;
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
                    update_ = true;
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
    quant_sync = true;
    return 0;
}
