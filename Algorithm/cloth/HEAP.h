///////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2002 - 2014, Huamin Wang
//  All rights reserved.
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//     1. Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//     2. Redistributions in binary form must reproduce the above copyright
//        notice, this list of conditions and the following disclaimer in the
//        documentation and/or other materials provided with the distribution.
//     3. The names of its contributors may not be used to endorse or promote
//        products derived from this software without specific prior written
//        permission.
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//	NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///////////////////////////////////////////////////////////////////////////////////////////
//  Class HEAP, small keys are put on the top.
///////////////////////////////////////////////////////////////////////////////////////////
//  Use an array to represent the binary tree. It is fast but needs more memory.
//  Need 2*n memory space at most. The tree must be balanced. n is the maximum 
//  cell number in the tree at one time.
///////////////////////////////////////////////////////////////////////////////////////////
#ifndef __MY_HEAP_H__
#define __MY_HEAP_H__
#include <stdio.h>


template <class TYPE>
class HEAP_NODE
{
public:
	float	key;
	TYPE	content;
	int		l_number;	//The number of children on the left branch
	int		r_number;	//The number of children on the right branch
};


template <class TYPE>
class HEAP
{
public:
	int	length;
	int	max_length;
	HEAP_NODE<TYPE>* data;
	
	HEAP(const int _max_length=1024)
	{		
		length		= 0;
		max_length	= _max_length;
		data		= new HEAP_NODE<TYPE>[max_length];
	}

	~HEAP()
	{
		if(data) delete[] data;
	}
    

///////////////////////////////////////////////////////////////////////////////////////////
//  Utility functions
///////////////////////////////////////////////////////////////////////////////////////////	
	bool Is_Empty()		{return length==0;}
	int Left(int i)		{return 2*i+1;}
	int Right(int i)	{return 2*i+2;}
	int Parent(int i)	{if(i==0) return -1; return (i-1)/2;}
	void Print()		{Print(0);}

	void Print(int i)
	{
		printf("id %d\n", data[i].content->id);
		if(data[i].l_number) { printf("%d's  left branch...\n", data[i].content->id); Print( Left(i));}
		if(data[i].r_number){ printf("%d's right branch...\n", data[i].content->id); Print(Right(i));}
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  Add a new element into the heap
///////////////////////////////////////////////////////////////////////////////////////////	
	int Add(TYPE content, float key)
	{
		length++;
		if(length==1) 
		{ 
			data[0].content	 = content; 
			data[0].key		 = key;
			data[0].l_number = 0;
			data[0].r_number = 0;
			return 0;
		}
		
		int current_node=0;
		while(1)
		{			
			if(current_node>=max_length) 
			{	
				//resize the array
				HEAP_NODE<TYPE>* new_data=new HEAP_NODE<TYPE>[2*max_length];
				memcpy(new_data, data, sizeof(HEAP_NODE<TYPE>)*max_length);
				delete[] data;
				data=new_data;			
				max_length=2*max_length;
			}
			
			if(data[current_node].l_number==0) 
			{
				data[current_node].l_number++; 
				current_node=Left(current_node);
				break;
			}
			if(data[current_node].r_number==0)
			{
				data[current_node].r_number++;
				current_node=Right(current_node); 
				break;
			}
			if(data[current_node].l_number<=data[current_node].r_number) 
			{
				data[current_node].l_number++;
				current_node=Left(current_node);
			}
			else
			{
				data[current_node].r_number++;
				current_node=Right(current_node);
			}
		}
		
		if(current_node>=max_length) 
		{	
			//resize the array
			HEAP_NODE<TYPE>* new_data=new HEAP_NODE<TYPE>[2*max_length];
			memcpy(new_data, data, sizeof(HEAP_NODE<TYPE>)*max_length);
			delete[] data;
			data=new_data;			
			max_length=2*max_length;
		}
		
		content->heap_node			= current_node;
		data[current_node].content	= content;
		data[current_node].key		= key;
		data[current_node].l_number	= 0;
		data[current_node].r_number	= 0;

		return Update_Node(current_node);
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  Remove or peek the top element of the heap (with the smallest key)
///////////////////////////////////////////////////////////////////////////////////////////
	TYPE Remove_Top()
	{
		if(length==0)	return 0;
		length--;
		TYPE result=data[0].content;
		if(data[0].l_number || data[0].r_number)	Fill_Hole(0);		
		return result;
	}

	TYPE Peek_Top()
	{
		return data[0].content;
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  Update the key of a node
///////////////////////////////////////////////////////////////////////////////////////////
	int Update_Key(int node, float key)
	{
		data[node].key=key;
		return Update_Node(node);
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  The functions needed to adjust the branch of a node
///////////////////////////////////////////////////////////////////////////////////////////
	int Update_Node(int node)
	{
		while(node>=0)
		{
			int switch_child=-1;
			if(data[node].l_number && data[node].key > data[ Left(node)].key) switch_child=Left(node);
			if(data[node].r_number && data[node].key > data[Right(node)].key)
			{
				if(switch_child==-1) switch_child=Right(node);
				else if(data[Right(node)].key<data[Left(node)].key) switch_child=Right(node);
			}
			if(switch_child==-1) break;

			Switch_Data(node, switch_child);
			node=switch_child;
		}
		while(Parent(node)>=0 && data[node].key<data[Parent(node)].key)
		{
			Switch_Data(node, Parent(node) );
			node=Parent(node);
		}
		return node;
	}

	void Switch_Data(int node1, int node2)
	{
		float temp_key		= data[node1].key;
		TYPE temp_content	= data[node1].content;
		data[node1].key		= data[node2].key;
		data[node1].content	= data[node2].content;
		data[node2].key		= temp_key;
		data[node2].content	= temp_content;

		data[node1].content->heap_node=node1;
		data[node2].content->heap_node=node2;
	}

	void Fill_Hole(int node)
	{
		while( data[node].l_number || data[node].r_number )
		{
			if((!data[node].r_number) || (data[node].l_number && data[node].r_number && data[Left(node)].key<=data[Right(node)].key))
			{
				data[node].key		= data[Left(node)].key;
				data[node].content	= data[Left(node)].content;
				data[node].content->heap_node=node;
				data[node].l_number--;
				node=Left(node);
				continue;
			}
			if((!data[node].l_number) || (data[node].l_number && data[node].r_number && data[Right(node)].key<=data[Left(node)].key))
			{
				data[node].key		= data[Right(node)].key;
				data[node].content	= data[Right(node)].content;
				data[node].content->heap_node=node;
				data[node].r_number--;
				node=Right(node);
				continue;
			}
		}
		if(Parent(node)>=0)
		{
			if(node%2==1) data[Parent(node)].l_number=0;
			if(node%2==0) data[Parent(node)].r_number=0;
		}
	}

};

#endif
