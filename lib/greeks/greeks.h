#ifndef GREEKS_H
#define GREEKS_H

#include <vector>
#include <string>

class greeks{
	
	private:
		int n;
		double err;
		
		std::string call;
		std::string put;
		
	public:
		greeks(int nn, double erp);
		
		double BS(std::vector<double> data, std::string opType);
		double Delta(std::vector<double> data, std::string opType);
		double Theta(std::vector<double> data, std::string opType);
		double Gamma(std::vector<double> data);
		double Vega(std::vector<double> data);
		double ImpliedVol(std::vector<double> data, std::string opType);
	
};



#endif