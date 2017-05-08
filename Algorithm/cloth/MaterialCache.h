#pragma once

#include <vector>
#include "ldpMat\ldp_basic_mat.h"
#include "cudpp\Cuda3DArray.h"
#include "cudpp\Cuda2DArray.h"
#include <map>
namespace ldp
{
	class MaterialCache
	{
	public:
		class StretchingData {
		public:
			enum {
				DIMS = 2,
				POINTS = 5
			};
			StretchingData(){
				d.resize(rows()*cols());
			}
			ldp::Float4& operator()(int x, int y){
				return d[y*cols() + x];
			}
			const ldp::Float4& operator()(int x, int y)const{
				return d[y*cols() + x];
			}
			ldp::Float4* data(){ return d.data(); }
			const ldp::Float4* data()const{ return d.data(); }
			int size()const{ return d.size(); }
			int rows()const{ return POINTS; }
			int cols()const{ return DIMS; }
		protected:
			std::vector<ldp::Float4> d;
		};

		class StretchingSamples {
		public:
			enum{
				SAMPLES = 30,
				SAMPLES2 = SAMPLES*SAMPLES,
			};
			StretchingSamples(){
				m_data.resize(SAMPLES*SAMPLES*SAMPLES);
				m_ary.create(make_int3(SAMPLES, SAMPLES, SAMPLES));
			}
			StretchingSamples(const StretchingSamples& rhs){
				m_data = rhs.m_data;
				m_ary = rhs.m_ary;
			}
			~StretchingSamples(){
				m_ary.release();
			}
			ldp::Float4& operator()(int x, int y, int z){
				return m_data[z*SAMPLES2 + y*SAMPLES + x];
			}
			const ldp::Float4& operator()(int x, int y, int z)const{
				return m_data[z*SAMPLES2 + y*SAMPLES + x];
			}
			ldp::Float4* data(){ return m_data.data(); }
			const ldp::Float4* data()const{ return m_data.data(); }
			int size()const{ return m_data.size(); }

			const Cuda3DArray<float4>& getCudaArray()const{ return m_ary; }
			Cuda3DArray<float4>& getCudaArray(){ return m_ary; }

			void updateHostToDevice(){ m_ary.fromHost((const float4*)m_data.data(), m_ary.size()); }
			void updateDeviceToHost(){ m_ary.toHost((float4*)m_data.data()); }
		protected:
			std::vector<ldp::Float4> m_data;
			Cuda3DArray<float4> m_ary;
		};

		class BendingData {
		public:
			enum {
				POINTS = 5,
				DIMS = 9,
				FilterMode = cudaFilterModePoint,
			};
			BendingData(){
				m_data.resize(rows()*cols());
				m_ary.create(make_int2(cols(), rows()), (cudaTextureFilterMode)FilterMode);
			}
			~BendingData(){
				m_ary.release();
			}
			float& operator()(int x, int y){
				return m_data[y*cols() + x];
			}
			const float& operator()(int x, int y)const{
				return m_data[y*cols() + x];
			}
			float* data(){ return m_data.data(); }
			const float* data()const{ return m_data.data(); }
			int size()const{ return m_data.size(); }
			int rows()const{ return POINTS; }
			int cols()const{ return DIMS; }

			const Cuda2DArray<float>& getCudaArray()const{ return m_ary; }
			Cuda2DArray<float>& getCudaArray(){ return m_ary; }

			void updateHostToDevice(){ m_ary.fromHost(data(), m_ary.size(), (cudaTextureFilterMode)FilterMode); }
			void updateDeviceToHost(){ m_ary.toHost(data()); }
		protected:
			std::vector<float> m_data;
			Cuda2DArray<float> m_ary;
		};
	
		struct Material
		{
			StretchingSamples stretchSample;
			BendingData bendData;
			float density;
		};
	public:
		MaterialCache();
		~MaterialCache();

		void clear();

		const Material* findMaterial(std::string name);
	protected:
		const Material* createMaterial(std::string name);
	protected:
		std::hash_map<std::string, std::shared_ptr<Material>> m_data;
	};
}