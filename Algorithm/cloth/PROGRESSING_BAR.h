//**************************************************************************************
// Copyright 2005 Huamin Wang.
//**************************************************************************************
// PROGRESSING_BAR classes
//**************************************************************************************
#ifndef __PROGRESSING_BAR_H__
#define __PROGRESSING_BAR_H__

#include "TIMER.h"
#include "stdio.h"


class PROGRESSING_BAR:public TIMER
{
public: 
	int percentage;
	int sample;
	int sample_number;

	PROGRESSING_BAR(double _sample_number)
		:sample(0), percentage(-1), sample_number(_sample_number)
	{}

	void Start(){sample=0;}
	
	void End(){printf("\n");}

	void Set(const double i)
	{
		sample=i;
		if( sample/(sample_number/100)> percentage)
		{
			percentage=sample/(sample_number/100);
			printf("\r[");
			for(int j=0; j<60; j++)
			{
				if( j*100/60<percentage) printf("+");
				else printf(" ");
			}
			printf("] [%3d%%] [%5.1fs]", percentage, Get_Time());
		}
	}

	void Update()
	{
		percentage=sample/(sample_number*0.01);
		printf("\r[");
		for(int j=0; j<60; j++)
		{
			if( j*100/60<percentage) printf("+");
			else printf(" ");
		}
		printf("] [%3d%%] [%5.1fs]", percentage, Get_Time());
	}

	bool Add(const int i=1)
	{
		sample+=i;
		if( (int)(sample/(sample_number*0.01))> percentage)
		{
			percentage=sample/(sample_number*0.01);
			printf("\r[");
			for(int j=0; j<60; j++)
			{
				if( j*100/60<percentage) printf("+");
				else printf(" ");
			}
			printf("] [%3d%%] [%5.1fs]", percentage, Get_Time());
			return 1;
		}
		return 0;
	}

};


#endif //__TIMER_H__