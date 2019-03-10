#define _USE_MATH_DEFINES
#include "greeks.h"

#include <iostream>
#include <vector>
#include <math.h>

greeks::greeks(int nn, double erp)
{	
	n = nn;
	err = erp; 
	
	call = "Call";
	put = "Put";
}

auto fpm = [](double vol, double pre, double bs, double vg){	
	double result = vol - ((bs - pre) / vg);
	return result;
};


auto adjust = [](std::vector<double> &data, std::string option){
	if(option == "Regular"){
		data[2] /= 100;
		data[3] /= 100;
		data[4] /= 12;
	} else {
		data[2] /= 100;
		data[3] /= 12;
	}
};

auto gauss = [](double x){
	double f = 1.0 / sqrt(2 * M_PI);
	double g = exp(-pow(x, 2) / 2);
	return f * g;
};

auto simps = [](int n){
	std::vector<double> r;
	double t;
	for(int i = 0; i < n; ++i)
	{
		if(i % 2 == 0)
		{
			t = 2;
		} else {
			t = 4;
		}
		if(i == 0 || i == n - 1)
		{
			t = 1;
		}
		r.push_back(t);
	}
	return r;
};

auto N = [](double x, int n){
	double a = -10;
	double dx = (x - a) / ((double) (n - 1));
	
	double result = 0;
	
	for(auto & t : simps(n))
	{
		result += gauss(a) * t;
		a += dx;
	}
	
	result *= (1.0/3.0) * dx;
	return result;
};

auto D = [](std::vector<double> data){
	double f = log(data[0]/data[1]) + (data[2] + pow(data[3],2)/2)*data[4];
	double d = data[3]*sqrt(data[4]);
	double d1 = f / d;
	std::vector<double> res = {d1, d1 - d};
	return res;
};

double greeks::Vega(std::vector<double> data)
{
	adjust(data, "Regular");
	double result = 0;
	std::vector<double> dd = D(data);
	
	result = (1.0 / 100.0) * data[0] * sqrt(data[4]) * gauss(dd[0]);
	
	return result;
}

double greeks::Gamma(std::vector<double> data)
{
	adjust(data, "Regular");
	double result = 0;
	std::vector<double> dd = D(data);
	
	double f = (1.0 / (data[0] * data[3] * sqrt(data[4])));
	result = f * gauss(dd[0]);
	
	return result;
}

double greeks::Theta(std::vector<double> data, std::string opType)
{
	adjust(data, "Regular");
	double result = 0;
	double ep = exp(-data[2]*data[4]);
	std::vector<double> dd = D(data);
	
	double a;
	double b;
	
	if(opType == call)
	{
		a = (-data[0]*data[3]) / (2 * sqrt(data[4]));
		b = data[2]*data[1]*ep*N(dd[1], n);
		result = a * gauss(dd[0]) - b;
	} else {
		a = (-data[0]*data[3]) / (2 * sqrt(data[4]));
		b = data[2]*data[1]*ep*N(-dd[1], n);
		result = a * gauss(dd[0]) + b;	
	}
	
	return result;
}

double greeks::Delta(std::vector<double> data, std::string opType)
{
	adjust(data, "Regular");
	double result = 0;
	std::vector<double> dd = D(data);
	
	if(opType == call)
	{
		result = N(dd[0], n);
	} else {
		result = N(dd[0], n) - 1;
	}
	
	return result;
}

double greeks::BS(std::vector<double> data, std::string opType)
{
	adjust(data, "Regular");
	double res = 0;
	double ep = exp(-data[2]*data[4]);
	std::vector<double> dd = D(data);
	
	if(opType == call)
	{
		res = data[0] * N(dd[0],n) - data[1] * ep * N(dd[1],n);
	} else {
		res = data[1] * ep * N(-dd[1],n) - data[0] * N(-dd[0],n);
	}
	
	return res;
}

double greeks::ImpliedVol(std::vector<double> data, std::string opType)
{
	double v0 = 5;
	double v1 = 0;
	
	double bs, vg;
	
	std::vector<double> temp;
	
	temp = {data[0],data[1],data[2],v0,data[3]};
	bs = BS(temp, opType);
	vg = Vega(temp);
	
	while(abs(v1 - v0) > err)
	{
		temp = {data[0],data[1],data[2],v0,data[3]};
		bs = BS(temp, opType);
		vg = Vega(temp);
		v1 = fpm(v0, data[4], bs, vg);
		v0 = v1;
		temp.clear();
	}
	
	
	return v1;
}

