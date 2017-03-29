//**************************************************************************************
// Copyright 2004 Huamin Wang.
//**************************************************************************************
// TIMER classes
//**************************************************************************************
#ifndef __TIMER_H__
#define __TIMER_H__

#include <sys/timeb.h>
#include <time.h>

class TIMER
{
public: 
	struct timeb start_time;

	TIMER(){Start();}

	~TIMER(){}

	void Start()
	{ftime( &start_time);}

	float Get_Time()
	{
		struct timeb current_time;
		ftime( &current_time);
		return (float)(current_time.time-start_time.time)+0.001f*(current_time.millitm-start_time.millitm);
	}
};


#endif //__TIMER_H__