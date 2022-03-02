#include <stdio.h>
#include <math.h>
#include <complex.h>

double _Complex slit_map(double c, double _Complex z)
{
	return (exp(c)/(2*z)) * (cpow(z,2) + 2*z*(1-exp(-c)) + 1 + cpow(z+1,2)*csqrt((cpow(z,2) + 2*z*(1-2*exp(-c))+1)/cpow(z+1,2)));
}

double _Complex building_block(double c, double _Complex z, double theta)
{
	return cexp(theta*I)*slit_map(c,cexp(-theta*I)*z);	
}

double _Complex full_map(double caps[], double _Complex z_0, double thetas[], int len)
{
	double _Complex z = z_0;
	int i;
	for(i=len-1; i >= 0; i = i-1)
	{
		z = building_block(caps[i],z,thetas[i]);
	}
	return z;
}

void full_map_py(double caps[], double z_real[], double z_imag[], double thetas[], int len_thetas, int len)
{
	int i;
	for(i=0;i<len;i=i+1)
	{
		double real = z_real[i];
		double imag = z_imag[i];
		double _Complex z = real + imag*I;
		double _Complex result = full_map(caps,z,thetas,len_thetas);
		z_real[i] = creal(result);
		z_imag[i] = cimag(result);
	}
}

double _Complex slit_diff(double c, double _Complex z)
{
	double _Complex sq_rt = csqrt((cpow(z,2) + 2*z*(1-2*exp(-c)) +1)/cpow(z+1,2));
	double _Complex prod1 = -(exp(c)/(2*cpow(z,2)))*(cpow(z,2) + 2*z*(1-exp(-c)) + 1 + cpow(z+1,2)*sq_rt);
	double _Complex prod2 = (exp(c)/(2*z))*(2*z + 2*(1-exp(-c)) + 2*(z+1)*sq_rt + 0.5*cpow(z+1,2)*(1/sq_rt)*(((2*z + 2*(1-2*exp(-c)))*cpow(z+1,2) - 2*(z+1)*(cpow(z,2) + 2*z*(1-2*exp(-c)) +1))/(cpow(z+1,4))));
	return prod1 + prod2;
}

void slit_diff_py(double c, double *z_real, double *z_imag)
{
	double real = *z_real;
	double imag = *z_imag;
	double _Complex z = real + imag*I;
	double _Complex result = slit_diff(c,z);
	*z_real = creal(result);
	*z_imag = cimag(result);
}

double _Complex map_diff(double _Complex z_0, double caps[], double thetas[], int len)
{
	double _Complex diff = 1;
	double _Complex z = z_0;
	int i;
	for(i=len-1;i>=0;i=i-1)
	{
		diff = diff*slit_diff(caps[i],cexp(-thetas[i]*I)*z);
		z = building_block(caps[i],z,thetas[i]);
	}
	return diff;
}

void map_diff_py(double *z_real, double *z_imag, double caps[], double thetas[], int len)
{
	double real = *z_real;
	double imag = *z_imag;
	double _Complex z = real + imag*I;
	double _Complex result = map_diff(z,caps,thetas,len);
	*z_real = creal(result);
	*z_imag = cimag(result);
}

double length(double c)
{
	return 2*exp(c)*(1 + sqrt(1-exp(-c))) - 2;	
}

double beta(double c)
{
	return 2*atan(length(c)/(2*sqrt(length(c)+1)));	
}

int generation(int m, double theta, double caps[], double angles[])
{
	int n;
	for(n=m;n>=0;n=n-1)
	{
		if(n==0)
		{
			return 0;
		}
		double angle = angles[n-1];
		double c = caps[n-1];
		double b = beta(c);
		if(angle-b<=theta && theta<=angle+b)
		{
			return n;
		}
		else if(angle -b < -M_PI && theta >= angle-b+M_PI)
		{
			return n;
		}
		else if(angle+b > M_PI && theta <= angle+b-M_PI)
		{
			return n;
		}
		theta = carg(building_block(c,cexp(theta*I),angle));
	}
}

double pdf(double theta, double sigma, double caps[], double thetas[], double eta, int len)
{
	return pow(cabs(map_diff(cexp(sigma+theta*I),caps,thetas,len)),-eta);	
}

void simpson(double vals[], double angles[], double sigma, double caps[], double thetas[], double eta, int len_thetas, int len)
{
	int i;
	for(i=0;i<len-1;i=i+1)
	{
		vals[i] = ((angles[i+1]-angles[i])/6)*(pdf(angles[i],sigma,caps,thetas,eta,len_thetas) + pdf(angles[i+1],sigma,caps,thetas,eta,len_thetas) + 4*pdf((angles[i]+angles[i+1])/2,sigma,caps,thetas,eta,len_thetas));	
	}
}

double new_cap(double c, double alpha, double sigma, double angle, double caps[], double thetas[], int len)
{
	return c*pow(cabs(map_diff(cexp(sigma+angle*I),caps,thetas,len)),-alpha);	
}

double choice(double choices[], double weights[], double total, double rand, int len)
{
	double num = 0;
	int i;
	for(i=0;i<len;i=i+1)
	{
		num = num + weights[i];
		if(rand < num)
		{
			return choices[i];
		}
	}
}