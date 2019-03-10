#ifndef COMPUTE_H
#define COMPUTE_H

#include <string>
#include <vector>

class Compute
{
	private:
	
		std::vector<double> holder;
	
	public:
		
		Compute(std::vector<std::string> data);

		double ask;
		double bid;
		double last;
		double rsi;
		bool rsi_sync;
		bool sync;
		
		std::string Call(std::string message);

};

#endif