#pragma once
#include <map>

namespace thrust_wrapper
{
	class cached_allocator
	{
	public:
		typedef char value_type;
		cached_allocator();
		~cached_allocator();
		char *allocate(std::ptrdiff_t num_bytes);
		void deallocate(char *ptr, size_t n);
	private:
		typedef std::multimap<std::ptrdiff_t, char*> free_blocks_type;
		typedef std::map<char *, std::ptrdiff_t>     allocated_blocks_type;
		free_blocks_type      free_blocks;
		allocated_blocks_type allocated_blocks;
		void free_all();
	};
	extern cached_allocator g_allocator;
}