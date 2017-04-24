#include "cached_allocator.h"
#include <algorithm>
#include <thrust\memory.h>
#include <thrust/system/cuda/vector.h>
namespace thrust_wrapper
{
	// Exponentially spaced buckets.
	static const int NumBuckets = 84;
	const size_t BucketSizes[NumBuckets] =
	{
		256, 512, 1024, 2048, 4096, 8192,
		12288, 16384, 24576, 32768, 49152, 65536,
		98304, 131072, 174848, 218624, 262144, 349696,
		436992, 524288, 655360, 786432, 917504, 1048576,
		1310720, 1572864, 1835008, 2097152, 2516736, 2936064,
		3355648, 3774976, 4194304, 4893440, 5592576, 6291456,
		6990592, 7689728, 8388608, 9786880, 11184896, 12582912,
		13981184, 15379200, 16777216, 18874368, 20971520, 23068672,
		25165824, 27262976, 29360128, 31457280, 33554432, 36910080,
		40265472, 43620864, 46976256, 50331648, 53687296, 57042688,
		60398080, 63753472, 67108864, 72701440, 78293760, 83886080,
		89478656, 95070976, 100663296, 106255872, 111848192, 117440512,
		123033088, 128625408, 134217728, 143804928, 153391872, 162978816,
		172565760, 182152704, 191739648, 201326592, 210913792, 220500736
	};

	static int LocateBucket(size_t size)
	{
		if (size > BucketSizes[NumBuckets - 1])
			return -1;

		return (int)(std::lower_bound(BucketSizes, BucketSizes + NumBuckets, size) -
			BucketSizes);
	}

	cached_allocator::cached_allocator() {}

	cached_allocator::~cached_allocator()
	{
		// free all allocations when cached_allocator goes out of scope
		free_all();
	}

	char *cached_allocator::allocate(std::ptrdiff_t num_bytes)
	{
		char *result = 0;


		int pos = LocateBucket(num_bytes);
		if (pos < 0 || pos >= NumBuckets)
			throw::std::exception("error: not supported size in thrust_wrapper::cached_allocator()");

		int nAllocate = BucketSizes[pos];

		// search the cache for a free block
		free_blocks_type::iterator free_block = free_blocks.find(nAllocate);

		if (free_block != free_blocks.end())
		{
			//std::cout << "cached_allocator::allocator(): found a hit" << std::endl;

			// get the pointer
			result = free_block->second;

			// erase from the free_blocks map
			free_blocks.erase(free_block);
		}
		else
		{
			// no allocation of the right size exists
			// create a new one with cuda::malloc
			// throw if cuda::malloc can't satisfy the request
			try
			{
				//std::cout << "cached_allocator::allocator(): no free block found; calling cuda::malloc" << std::endl;

				// allocate memory and convert cuda::pointer to raw pointer
				result = thrust::cuda::malloc<char>(nAllocate).get();
			}
			catch (std::runtime_error &e)
			{
				throw;
			}
		}

		// insert the allocated pointer into the allocated_blocks map
		allocated_blocks.insert(std::make_pair(result, nAllocate));

		return result;
	}

	void cached_allocator::deallocate(char *ptr, size_t n)
	{
		// erase the allocated block from the allocated blocks map
		allocated_blocks_type::iterator iter = allocated_blocks.find(ptr);
		std::ptrdiff_t num_bytes = iter->second;
		allocated_blocks.erase(iter);

		// insert the block into the free blocks map
		free_blocks.insert(std::make_pair(num_bytes, ptr));
	}

	void cached_allocator::free_all()
	{
		std::cout << "cached_allocator::free_all(): cleaning up after ourselves..." << std::endl;

		// deallocate all outstanding blocks in both lists
		for (free_blocks_type::iterator i = free_blocks.begin();
			i != free_blocks.end();
			++i)
		{
			// transform the pointer to cuda::pointer before calling cuda::free
			thrust::cuda::free(thrust::cuda::pointer<char>(i->second));
		}

		for (allocated_blocks_type::iterator i = allocated_blocks.begin();
			i != allocated_blocks.end();
			++i)
		{
			// transform the pointer to cuda::pointer before calling cuda::free
			thrust::cuda::free(thrust::cuda::pointer<char>(i->first));
		}
	}

	cached_allocator g_allocator;
}