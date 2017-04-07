extern "C" texture<int,1,cudaReadModeElementType> __tex0;
extern "C" texture<int,1,cudaReadModeElementType> __tex1;
extern "C" texture<int,1,cudaReadModeElementType> __tex2;
extern "C" texture<int,1,cudaReadModeElementType> __tex3;
struct g_cu_a;
struct pfm;
struct dlist_float;
struct CUarray_t;
struct srbf;
struct g_cu_u;
struct g_cu__;
struct g_cu_fc;
struct g_cu_ab;
struct g_cu_c;
struct g_cu_m;
struct g_cu_ac;
struct g_cu_d;
struct g_cu_qd;
struct g_cu_tb;
struct g_cu_fcgc;
struct g_cu_nb;
struct g_cu_de;
struct g_cu_t;
typedef void(*void_Func_void8)(void*);
struct pfm{void* __vftab;
int __refcnt;
void* __free;
int _w;
int _h;
int _nc;
float* _raster;
struct dlist_float* _rasterD;
struct dlist_float* _approxD;
struct CUarray_t* _ta;
};
struct srbf{float2 _center;
float3 _weight;
float _lambda;
};
struct g_cu_a{int __thread_size;
struct pfm m_b;
char __pad[4]; struct srbf m_c;
float* d;
char __final_pad[4]; };
struct g_cu_u{int nbgred;
int n;
float* dd;
float* b;
};
struct g_cu__{int __thread_size;
struct pfm m_ab;
int m_bb;
float div;
float* tmp;
void* m_cb;
char* d;
};
struct g_cu_fc{int __thread_size;
void* m_gc;
char* d;
float* tmp;
};
struct g_cu_ab{int __thread_size;
};
struct g_cu_c{int __thread_size;
int m_d;
struct srbf* m_e;
void* st;
int m_f;
};
struct g_cu_m{int __thread_size;
int m_n;
struct pfm m_o;
float div;
float* tmp;
};
struct g_cu_ac{int __thread_size;
};
struct g_cu_d{int __thread_size;
int m_e;
struct pfm m_f;
float div;
int i;
int m_g;
void* st;
int m_h;
float* tmp;
void* m_k;
char* d;
int SPAPstride;
};
struct g_cu_qd{int __thread_size;
void* m_rd;
char* d;
int SPAPstride;
int i;
float* m_sd;
float ret;
float* tmp;
};
struct g_cu_tb{int __thread_size;
void* m_ub;
char* d;
int SPAPstride;
int n;
int i;
float* m_vb;
float ret;
float* tmp;
};
struct g_cu_fcgc{int __thread_size;
void* m_hc;
char* d;
int SPAPstride;
int n;
int i;
float* m_kc;
float ret;
float* tmp;
float* m_lc;
float* m_mc;
};
struct g_cu_nb{int __thread_size;
int lambda_begin;
int i;
float* d;
float ret;
int centers_begin;
float m_ob;
int n;
float m_pb;
};
struct g_cu_de{int __thread_size;
int width;
int height;
int nc;
float* pbo_data;
};
struct g_cu_t{int __thread_size;
int width;
int height;
int nc;
float* pbo_data;
};
typedef int carray_int_3[3];
struct dlist_float{void* __vftab;
int __refcnt;
void* __free;
void* _d;
int _n;
int _sz;
carray_int_3* elements;
};
struct CUarray_t{};
extern "C" __global__ void CUDAkrnl_102(struct g_cu_a __param);
extern "C" __global__ void CUDAkrnl_1588(struct g_cu_u __param);
extern "C" __global__ void CUDAkrnl_571(struct g_cu__ __param);
extern "C" __global__ void CUDAkrnl_587(struct g_cu_fc __param);
extern "C" __global__ void CUDAkrnl_588(struct g_cu_ab __param);
extern "C" __global__ void CUDAkrnl_593(struct g_cu_c __param);
extern "C" __global__ void CUDAkrnl_655(struct g_cu_m __param);
extern "C" __global__ void CUDAkrnl_670(struct g_cu_ac __param);
extern "C" __global__ void CUDAkrnl_674(struct g_cu_d __param);
extern "C" __global__ void CUDAkrnl_697(struct g_cu_qd __param);
extern "C" __global__ void CUDAkrnl_697a(struct g_cu_tb __param);
extern "C" __global__ void CUDAkrnl_697b(struct g_cu_fcgc __param);
extern "C" __global__ void CUDAkrnl_703(struct g_cu_nb __param);
extern "C" __global__ void CUDAkrnl_749(struct g_cu_de __param);
extern "C" __global__ void CUDAkrnl_835(struct g_cu_t __param);


#line 102 "d:\\projects\\HairRenderer\\HairRenderer\\Env\\SRBFGen_src\\tool.i"
extern "C" __global__ void CUDAkrnl_102(struct g_cu_a __param){int t_a;int x;int y;float3* thisb;int2 t_c;float t;float b;float td;float a;float2 t_e;float2 t_f;float lambda;float t_g;float* t_h;float bk;float t_l;float t_m;float t_n;int ao;float* t_p;int aq;float* t_r;int as;float* t_t;t_a=blockIdx.x*blockDim.x+threadIdx.x;if(!((t_a<__param.__thread_size))){return;}

	x=(t_a%((&__param.m_b))->_w);
	y=(t_a/ ((&__param.m_b))->_w);thisb=(&((&__param.m_c))->_weight);
	t_c=make_int2(x,y);t=((float)(t_c).y/ (float)((&__param.m_b))->_h);b=((t*1.80000000e+002f)+ -9.00000000e+001f);td=((float)(t_c).x/ (float)((&__param.m_b))->_w);a=(td*3.60000000e+002f);t_e=make_float2(a,b);t_f=make_float2(((t_e).x*1.74532905e-002f),((t_e).y*1.74532905e-002f));lambda=((&__param.m_c))->_lambda;t_g=(sinf((t_f).y)*sinf(((&__param.m_c))->_center.y));t_h=(float*)((&__param.m_c));bk=exp((((((cosf((t_f).y)*cosf(((&__param.m_c))->_center.y))*cosf(((t_f).x- *t_h)))+ t_g)- 1.00000000e+000f)/ lambda));t_l=(bk*thisb->x);t_m=(bk*thisb->y);t_n=(bk*thisb->z);
	ao=(t_a*((&__param.m_b))->_nc);t_p=(__param.d + ao);*t_p=(*t_p+ t_l);
	aq=((t_a*((&__param.m_b))->_nc)+ 1);t_r=(__param.d + aq);*t_r=(*t_r+ t_m);
	as=((t_a*((&__param.m_b))->_nc)+ 2);t_t=(__param.d + as);*t_t=(*t_t+ t_n);}

#line 1588 "D:\\projects\\kara\\units\\cubsgp.i"
extern "C" __global__ void CUDAkrnl_1588(struct g_cu_u __param){int t_a;float c;int t_b;int t_c;int i;float y;void* t_d;int ie;float* a;volatile float* af;float yg;float x;float yh;float xk;float yl;float xm;float yo;float xp;float yq;float xr;float ys;float xt;float yu;float xv;float yw;float xx;float t_y;int t_z;t_a=blockIdx.x*blockDim.x+threadIdx.x;__shared__ int __sharedbase[256];
#line 1592 "D:\\projects\\kara\\units\\cubsgp.i"
	c=__int_as_float(0);
	t_b=(__param.nbgred<<8);t_c=(__param.n- 1);i=t_a;for(;;){if(!((i<=t_c))){break;}
		y=(__param.dd)[i];c=(c+ y);
#line 1593 "D:\\projects\\kara\\units\\cubsgp.i"
		i=(i+ t_b);}t_d=(void*)((char*)(void*)__sharedbase+0);


	ie=threadIdx.x;;

	a=(float*)(((char*)(t_d)+ (ie<<2)));af=a;
	*af=c;__syncthreads();

	if((ie<128)){yg=(af)[128];x=*af;*af=(x+ yg);}__syncthreads();

	if((ie<64)){yh=(af)[64];xk=*af;*af=(xk+ yh);}__syncthreads();

	if(!((ie<32))){return;}
	yl=(af)[32];xm=*af;*af=(xm+ yl);
	yo=(af)[16];xp=*af;*af=(xp+ yo);
	yq=(af)[8];xr=*af;*af=(xr+ yq);
	ys=(af)[4];xt=*af;*af=(xt+ ys);
	yu=(af)[2];xv=*af;*af=(xv+ yu);
	if(!((ie==0))){return;}
	yw=(af)[1];xx=*af;t_y=(xx+ yw);t_z=blockIdx.x;;(__param.b)[t_z]=t_y;}

#line 571 "d:\\projects\\HairRenderer\\HairRenderer\\Env\\SRBFGen_src\\tool.i"
extern "C" __global__ void CUDAkrnl_571(struct g_cu__ __param){int t_a;int b;int a;int2 t_b;float t;float bc;float td;float ae;float2 t_f;float2 t_g;int bh;int ak;int2 t_l;float tm;float bn;float to;float ap;float2 t_q;float x;float texcoordx;float t_r;float ts;float xt;float texcoordy;float t_u;float tv;float4 rr;float4* t_w;float4 t_x;float4 t_y;float4 t_z;float cos_theta;int ii;float4 rr_;float t_ab;float4 rrbb;float t_cb;float4 rrdb;float err;float4 rreb;float t_fb;float4 rrgb;float base;int iihb;float4 rrkb;float t_lb;float4 rrmb;float t_nb;float4 rrob;float errpb;float4 rrqb;float t_rb;float4 rrsb;float basetb;int iiub;float4 rrvb;float t_wb;float4 rrxb;float t_yb;float4 rrzb;float err_b;float4 rrac;float t_bc;float4 rrcc;float basedc;float* t_ec;t_a=blockIdx.x*blockDim.x+threadIdx.x;if(!((t_a<__param.__thread_size))){return;}

	b=(t_a/ ((&__param.m_ab))->_w);a=(t_a%((&__param.m_ab))->_w);t_b=make_int2(a,b);t=((float)(t_b).y/ (float)((&__param.m_ab))->_h);bc=((t*1.80000000e+002f)+ -9.00000000e+001f);td=((float)(t_b).x/ (float)((&__param.m_ab))->_w);ae=(td*3.60000000e+002f);t_f=make_float2(ae,bc);t_g=make_float2(((t_f).x*1.74532905e-002f),((t_f).y*1.74532905e-002f));
	if((__param.m_bb!=0)){
		bh=(t_a/ ((&__param.m_ab))->_w);ak=(t_a%((&__param.m_ab))->_w);t_l=make_int2(ak,bh);
		tm=((float)(t_l).y/ (float)((&__param.m_ab))->_h);bn=((tm*3.14159226e+000f)+ -1.57079613e+000f);to=((float)(t_l).x/ (float)((&__param.m_ab))->_w);ap=(to*6.28318453e+000f);t_q=make_float2(ap,bn);
		x=((t_q).x/ 6.28318453e+000f);if((x<1.53000012e-001f)){texcoordx=__int_as_float(0);}else{if((x>3.97000015e-001f)){t_r=1.00000000e+000f;}else{ts=((x- 1.53000012e-001f)/ 2.44000003e-001f);t_r=ts;}texcoordx=t_r;}xt=(((t_q).y+ 1.57079613e+000f)/ 3.14159226e+000f);if((xt<3.23000014e-001f)){texcoordy=__int_as_float(0);}else{if((xt>7.73000002e-001f)){t_u=1.00000000e+000f;}else{tv=((xt- 3.23000014e-001f)/ 4.49999988e-001f);t_u=tv;}texcoordy=t_u;}rr=tex2D(*(texture<float4,2,cudaReadModeElementType>*)&__tex2,texcoordx,texcoordy);;t_w=&rr;t_x=*t_w;t_y=make_float4((t_x).x,(t_x).y,(t_x).z,(t_x).w);t_z=t_y;}
	cos_theta=cosf((t_g).y);
#line 583 "d:\\projects\\HairRenderer\\HairRenderer\\Env\\SRBFGen_src\\tool.i"
	ii=(t_a*3);
	if((__param.m_bb!=0)){rr_=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,ii);;t_ab=((t_z).x- rr_.x);}else{rrbb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,ii);;t_cb=rrbb.x;rrdb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,ii);;t_ab=(rrdb.x- t_cb);}
	err=(((t_ab*cos_theta)*__param.div)*t_ab);
	rreb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,ii);;t_fb=rreb.x;rrgb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,ii);;base=(((rrgb.x*cos_theta)*__param.div)*t_fb);
#line 583 "d:\\projects\\HairRenderer\\HairRenderer\\Env\\SRBFGen_src\\tool.i"
	iihb=((t_a*3)+ 1);
	if((__param.m_bb!=0)){rrkb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,iihb);;t_lb=((t_z).y- rrkb.x);}else{rrmb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,iihb);;t_nb=rrmb.x;rrob=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,iihb);;t_lb=(rrob.x- t_nb);}
	errpb=(err+ (((t_lb*cos_theta)*__param.div)*t_lb));
	rrqb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,iihb);;t_rb=rrqb.x;rrsb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,iihb);;basetb=(base+ (((rrsb.x*cos_theta)*__param.div)*t_rb));
#line 583 "d:\\projects\\HairRenderer\\HairRenderer\\Env\\SRBFGen_src\\tool.i"
	iiub=((t_a*3)+ 2);
	if((__param.m_bb!=0)){rrvb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,iiub);;t_wb=((t_z).z- rrvb.x);}else{rrxb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,iiub);;t_yb=rrxb.x;rrzb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,iiub);;t_wb=(rrzb.x- t_yb);}
	err_b=(errpb+ (((t_wb*cos_theta)*__param.div)*t_wb));
	rrac=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,iiub);;t_bc=rrac.x;rrcc=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,iiub);;basedc=(basetb+ (((rrcc.x*cos_theta)*__param.div)*t_bc));
	(__param.tmp)[t_a]=err_b;t_ec=(float*)((__param.d + (t_a<<2)));*t_ec=basedc;}

#line 587 "d:\\projects\\HairRenderer\\HairRenderer\\Env\\SRBFGen_src\\tool.i"
extern "C" __global__ void CUDAkrnl_587(struct g_cu_fc __param){int t_a;float* t_b;float base;t_a=blockIdx.x*blockDim.x+threadIdx.x;if(!((t_a<__param.__thread_size))){return;}t_b=(float*)((__param.d + (t_a<<2)));base=*t_b;
	(__param.tmp)[t_a]=base;}

#line 588 "d:\\projects\\HairRenderer\\HairRenderer\\Env\\SRBFGen_src\\tool.i"
extern "C" __global__ void CUDAkrnl_588(struct g_cu_ab __param){}

#line 593 "d:\\projects\\HairRenderer\\HairRenderer\\Env\\SRBFGen_src\\tool.i"
extern "C" __global__ void CUDAkrnl_593(struct g_cu_c __param){int d;int da;struct srbf* t_b;char* ptr;int stride;float t_c;float* t_d;float t_e;float* t_f;float t_g;float* t_h;float t_k;float* t_l;d=blockIdx.x*blockDim.x+threadIdx.x;if(!((d<__param.__thread_size))){return;}

	da=(d+ __param.m_d);t_b=(__param.m_e + d);ptr=*(char**)((&__param.st));stride=__param.m_f;memcpy((void*)((ptr + (da<<3))),(void*)(t_b),8);t_c=t_b->_weight.x;t_d=(float*)((ptr + ((da+ (stride<<1))<<2)));*t_d=t_c;t_e=t_b->_weight.y;t_f=(float*)((ptr + ((da+ (stride*3))<<2)));*t_f=t_e;t_g=t_b->_weight.z;t_h=(float*)((ptr + ((da+ (stride<<2))<<2)));*t_h=t_g;t_k=t_b->_lambda;t_l=(float*)((ptr + ((da+ (stride*5))<<2)));*t_l=t_k;}

#line 655 "d:\\projects\\HairRenderer\\HairRenderer\\Env\\SRBFGen_src\\tool.i"
extern "C" __global__ void CUDAkrnl_655(struct g_cu_m __param){int t_a;int b;int a;int2 t_b;float t;float bc;float td;float ae;float2 t_f;float x;float texcoordx;float t_g;float th;float xk;float texcoordy;float t_l;float tm;float4 rr;float4* t_n;float4 t_o;float4 t_p;float4 t_q;int br;int as;int2 t_t;float tu;float bv;float tw;float ax;float2 t_y;float2 t_z;float cos_theta;int ii;float4 rr_;float t_ab;float4 rrbb;float t_cb;float4 rrdb;float err;int iieb;float4 rrfb;float t_gb;float4 rrhb;float t_kb;float4 rrlb;float errmb;int iinb;float4 rrob;float t_pb;float4 rrqb;float t_rb;float4 rrsb;float errtb;t_a=blockIdx.x*blockDim.x+threadIdx.x;if(!((t_a<__param.__thread_size))){return;}

	if((__param.m_n!=0)){
		b=(t_a/ ((&__param.m_o))->_w);a=(t_a%((&__param.m_o))->_w);t_b=make_int2(a,b);
		t=((float)(t_b).y/ (float)((&__param.m_o))->_h);bc=((t*3.14159226e+000f)+ -1.57079613e+000f);td=((float)(t_b).x/ (float)((&__param.m_o))->_w);ae=(td*6.28318453e+000f);t_f=make_float2(ae,bc);
		x=((t_f).x/ 6.28318453e+000f);if((x<1.53000012e-001f)){texcoordx=__int_as_float(0);}else{if((x>3.97000015e-001f)){t_g=1.00000000e+000f;}else{th=((x- 1.53000012e-001f)/ 2.44000003e-001f);t_g=th;}texcoordx=t_g;}xk=(((t_f).y+ 1.57079613e+000f)/ 3.14159226e+000f);if((xk<3.23000014e-001f)){texcoordy=__int_as_float(0);}else{if((xk>7.73000002e-001f)){t_l=1.00000000e+000f;}else{tm=((xk- 3.23000014e-001f)/ 4.49999988e-001f);t_l=tm;}texcoordy=t_l;}rr=tex2D(*(texture<float4,2,cudaReadModeElementType>*)&__tex2,texcoordx,texcoordy);;t_n=&rr;t_o=*t_n;t_p=make_float4((t_o).x,(t_o).y,(t_o).z,(t_o).w);t_q=t_p;}
	br=(t_a/ ((&__param.m_o))->_w);as=(t_a%((&__param.m_o))->_w);t_t=make_int2(as,br);tu=((float)(t_t).y/ (float)((&__param.m_o))->_h);bv=((tu*1.80000000e+002f)+ -9.00000000e+001f);tw=((float)(t_t).x/ (float)((&__param.m_o))->_w);ax=(tw*3.60000000e+002f);t_y=make_float2(ax,bv);t_z=make_float2(((t_y).x*1.74532905e-002f),((t_y).y*1.74532905e-002f));
	cos_theta=cosf((t_z).y);
#line 666 "d:\\projects\\HairRenderer\\HairRenderer\\Env\\SRBFGen_src\\tool.i"
	ii=(t_a*3);
	if((__param.m_n!=0)){rr_=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,ii);;t_ab=((t_q).x- rr_.x);}else{rrbb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,ii);;t_cb=rrbb.x;rrdb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,ii);;t_ab=(rrdb.x- t_cb);}
	err=(((t_ab*cos_theta)*t_ab)*__param.div);
#line 666 "d:\\projects\\HairRenderer\\HairRenderer\\Env\\SRBFGen_src\\tool.i"
	iieb=((t_a*3)+ 1);
	if((__param.m_n!=0)){rrfb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,iieb);;t_gb=((t_q).y- rrfb.x);}else{rrhb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,iieb);;t_kb=rrhb.x;rrlb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,iieb);;t_gb=(rrlb.x- t_kb);}
	errmb=(err+ (((t_gb*cos_theta)*t_gb)*__param.div));
#line 666 "d:\\projects\\HairRenderer\\HairRenderer\\Env\\SRBFGen_src\\tool.i"
	iinb=((t_a*3)+ 2);
	if((__param.m_n!=0)){rrob=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,iinb);;t_pb=((t_q).z- rrob.x);}else{rrqb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,iinb);;t_rb=rrqb.x;rrsb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,iinb);;t_pb=(rrsb.x- t_rb);}
	errtb=(errmb+ (((t_pb*cos_theta)*t_pb)*__param.div));

	(__param.tmp)[t_a]=errtb;}

#line 670 "d:\\projects\\HairRenderer\\HairRenderer\\Env\\SRBFGen_src\\tool.i"
extern "C" __global__ void CUDAkrnl_670(struct g_cu_ac __param){}



extern "C" __global__ void CUDAkrnl_674(struct g_cu_d __param){int t_a;int b;int a;int2 t_b;float t;float bc;float td;float ae;float2 t_f;float x;float texcoordx;float t_g;float th;float xk;float texcoordy;float t_l;float tm;float4 rr;float4* t_n;float4 t_o;float4 t_p;float4 t_q;int br;int as;int2 t_t;float tu;float bv;float tw;float ax;float2 t_y;float2 t_z;float cos_theta;int ii;float4 rr_;float aab;float4 rrbb;float t_cb;float4 rrdb;float aeb;int iifb;float4 rrgb;float ahb;float4 rrkb;float t_lb;float4 rrmb;float anb;int iiob;float4 rrpb;float aqb;float4 rrrb;float t_sb;float4 rrtb;float aub;int d;int dvb;float2* t_wb;float2 t_xb;int dyb;int dzb;char* t__b;int t_ac;float* t_bc;float t_cc;float* t_dc;float t_ec;float* t_fc;float t_gc;float bhc;float4* t_kc;float4 rrlc;int amc;float c;float dEdwi;float2* t_nc;int t_oc;int apc;float2* t_qc;int t_rc;int asc;float* t_tc;int t_uc;int avc;float* t_wc;int t_xc;int ayc;float* t_zc;int t__c;int aad;float* t_bd;int t_cd;int add;float* t_ed;int t_fd;int agd;float* t_hd;int t_kd;int ald;float* t_md;int t_nd;int aod;float* t_pd;t_a=blockIdx.x*blockDim.x+threadIdx.x;if(!((t_a<__param.__thread_size))){return;}

	if((__param.m_e!=0)){
		b=(t_a/ ((&__param.m_f))->_w);a=(t_a%((&__param.m_f))->_w);t_b=make_int2(a,b);
		t=((float)(t_b).y/ (float)((&__param.m_f))->_h);bc=((t*3.14159226e+000f)+ -1.57079613e+000f);td=((float)(t_b).x/ (float)((&__param.m_f))->_w);ae=(td*6.28318453e+000f);t_f=make_float2(ae,bc);
		x=((t_f).x/ 6.28318453e+000f);if((x<1.53000012e-001f)){texcoordx=__int_as_float(0);}else{if((x>3.97000015e-001f)){t_g=1.00000000e+000f;}else{th=((x- 1.53000012e-001f)/ 2.44000003e-001f);t_g=th;}texcoordx=t_g;}xk=(((t_f).y+ 1.57079613e+000f)/ 3.14159226e+000f);if((xk<3.23000014e-001f)){texcoordy=__int_as_float(0);}else{if((xk>7.73000002e-001f)){t_l=1.00000000e+000f;}else{tm=((xk- 3.23000014e-001f)/ 4.49999988e-001f);t_l=tm;}texcoordy=t_l;}rr=tex2D(*(texture<float4,2,cudaReadModeElementType>*)&__tex3,texcoordx,texcoordy);;t_n=&rr;t_o=*t_n;t_p=make_float4((t_o).x,(t_o).y,(t_o).z,(t_o).w);t_q=t_p;}
	br=(t_a/ ((&__param.m_f))->_w);as=(t_a%((&__param.m_f))->_w);t_t=make_int2(as,br);tu=((float)(t_t).y/ (float)((&__param.m_f))->_h);bv=((tu*1.80000000e+002f)+ -9.00000000e+001f);tw=((float)(t_t).x/ (float)((&__param.m_f))->_w);ax=(tw*3.60000000e+002f);t_y=make_float2(ax,bv);t_z=make_float2(((t_y).x*1.74532905e-002f),((t_y).y*1.74532905e-002f));
	cos_theta=cosf((t_z).y);


	ii=(t_a*3);
	if((__param.m_e!=0)){rr_=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,ii);;aab=((t_q).x- rr_.x);}else{rrbb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,ii);;t_cb=rrbb.x;rrdb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,ii);;aab=(rrdb.x- t_cb);}
	aeb=(aab*((-cos_theta*__param.div)*2.00000000e+000f));
#line 684 "d:\\projects\\HairRenderer\\HairRenderer\\Env\\SRBFGen_src\\tool.i"
	iifb=((t_a*3)+ 1);
	if((__param.m_e!=0)){rrgb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,iifb);;ahb=((t_q).y- rrgb.x);}else{rrkb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,iifb);;t_lb=rrkb.x;rrmb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,iifb);;ahb=(rrmb.x- t_lb);}
	anb=(ahb*((-cos_theta*__param.div)*2.00000000e+000f));
#line 684 "d:\\projects\\HairRenderer\\HairRenderer\\Env\\SRBFGen_src\\tool.i"
	iiob=((t_a*3)+ 2);
	if((__param.m_e!=0)){rrpb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,iiob);;aqb=((t_q).z- rrpb.x);}else{rrrb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,iiob);;t_sb=rrrb.x;rrtb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,iiob);;aqb=(rrtb.x- t_sb);}
	aub=(aqb*((-cos_theta*__param.div)*2.00000000e+000f));d=__param.i;
	dvb=(d+ __param.m_g);t_wb=(float2*)((*(char**)((&__param.st)) + (dvb<<3)));t_xb=*t_wb;dyb=__param.i;
	dzb=(dyb+ __param.m_g);t__b=*(char**)((&__param.st));t_ac=__param.m_h;t_bc=(float*)((t__b + ((dzb+ (t_ac<<1))<<2)));t_cc=*t_bc;t_dc=(float*)((t__b + ((dzb+ (t_ac*3))<<2)));t_ec=*t_dc;t_fc=(float*)((t__b + ((dzb+ (t_ac<<2))<<2)));t_gc=*t_fc;
	bhc=((((cosf((t_z).y)*cosf((t_xb).y))*cosf(((t_z).x- (t_xb).x)))+ (sinf((t_z).y)*sinf((t_xb).y)))- 1.00000000e+000f);
	t_kc=&rrlc;amc=__param.i;*t_kc=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex2,amc);;c=exp((bhc/ rrlc.x));
#line 696 "d:\\projects\\HairRenderer\\HairRenderer\\Env\\SRBFGen_src\\tool.i"
	dEdwi=(aeb*c);
	(__param.tmp)[t_a]=dEdwi;t_nc=(float2*)((__param.d + (t_a<<3)));*t_nc=t_xb;t_oc=(t_a<<3);apc=(__param.SPAPstride<<3);t_qc=(float2*)(((__param.d + apc) + t_oc));*t_qc=t_z;t_rc=(t_a<<2);asc=(__param.SPAPstride<<4);t_tc=(float*)(((__param.d + asc) + t_rc));*t_tc=t_cc;t_uc=(t_a<<2);avc=(__param.SPAPstride*20);t_wc=(float*)(((__param.d + avc) + t_uc));*t_wc=dEdwi;t_xc=(t_a<<2);ayc=(__param.SPAPstride*24);t_zc=(float*)(((__param.d + ayc) + t_xc));*t_zc=anb;t__c=(t_a<<2);aad=(__param.SPAPstride<<5);t_bd=(float*)(((__param.d + aad) + t__c));*t_bd=t_gc;t_cd=(t_a<<2);add=(__param.SPAPstride*36);t_ed=(float*)(((__param.d + add) + t_cd));*t_ed=bhc;t_fd=(t_a<<2);agd=(__param.SPAPstride*40);t_hd=(float*)(((__param.d + agd) + t_fd));*t_hd=t_ec;t_kd=(t_a<<2);ald=(__param.SPAPstride*44);t_md=(float*)(((__param.d + ald) + t_kd));*t_md=aub;t_nd=(t_a<<2);aod=(__param.SPAPstride*48);t_pd=(float*)(((__param.d + aod) + t_nd));*t_pd=c;}

#line 697 "d:\\projects\\HairRenderer\\HairRenderer\\Env\\SRBFGen_src\\tool.i"
extern "C" __global__ void CUDAkrnl_697(struct g_cu_qd __param){int t_a;float2* t_b;float2 t_c;int t_d;int a;float2* t_e;float2 t_f;int t_g;int ah;float* t_k;float t_l;int t_m;int an;float* t_o;float dEdwi;int t_p;int aq;float* t_r;float as;int t_t;int au;float* t_v;float b;int t_w;int ax;float* t_y;float c;int az;float4* t__;float4 rr;int aab;float d;float4* t_bb;float4 rrcb;int adb;float dEdli;float dEdcx;float dEdcy;float dEdwieb;int t_fb;int agb;float* t_hb;int t_kb;int alb;float* t_mb;int t_nb;int aob;float* t_pb;int t_qb;int arb;float* t_sb;t_a=blockIdx.x*blockDim.x+threadIdx.x;if(!((t_a<__param.__thread_size))){return;}t_b=(float2*)((__param.d + (t_a<<3)));t_c=*t_b;t_d=(t_a<<3);a=(__param.SPAPstride<<3);t_e=(float2*)(((__param.d + a) + t_d));t_f=*t_e;t_g=(t_a<<2);ah=(__param.SPAPstride<<4);t_k=(float*)(((__param.d + ah) + t_g));t_l=*t_k;t_m=(t_a<<2);an=(__param.SPAPstride*20);t_o=(float*)(((__param.d + an) + t_m));dEdwi=*t_o;t_p=(t_a<<2);aq=(__param.SPAPstride*24);t_r=(float*)(((__param.d + aq) + t_p));as=*t_r;t_t=(t_a<<2);au=(__param.SPAPstride*36);t_v=(float*)(((__param.d + au) + t_t));b=*t_v;t_w=(t_a<<2);ax=(__param.SPAPstride*48);t_y=(float*)(((__param.d + ax) + t_w));c=*t_y;az=__param.i;(__param.m_sd)[az]=__param.ret;
	t__=&rr;aab=__param.i;*t__=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,aab);;d=((dEdwi*t_l)/ rr.x);
	t_bb=&rrcb;adb=__param.i;*t_bb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,adb);;dEdli=((-d*b)/ rrcb.x);
	dEdcx=(d*((-cosf((t_c).y)*cosf((t_f).y))*sinf(((t_c).x- (t_f).x))));
	dEdcy=(d*((cosf((t_c).y)*sinf((t_f).y))- ((sinf((t_c).y)*cosf((t_f).y))*cosf(((t_c).x- (t_f).x)))));
#line 696 "d:\\projects\\HairRenderer\\HairRenderer\\Env\\SRBFGen_src\\tool.i"
	dEdwieb=(as*c);
	(__param.tmp)[t_a]=dEdwieb;t_fb=(t_a<<2);agb=(__param.SPAPstride<<4);t_hb=(float*)(((__param.d + agb) + t_fb));*t_hb=dEdwieb;t_kb=(t_a<<2);alb=(__param.SPAPstride*20);t_mb=(float*)(((__param.d + alb) + t_kb));*t_mb=dEdli;t_nb=(t_a<<2);aob=(__param.SPAPstride*24);t_pb=(float*)(((__param.d + aob) + t_nb));*t_pb=dEdcx;t_qb=(t_a<<2);arb=(__param.SPAPstride*28);t_sb=(float*)(((__param.d + arb) + t_qb));*t_sb=dEdcy;}

#line 697 "d:\\projects\\HairRenderer\\HairRenderer\\Env\\SRBFGen_src\\tool.i"
extern "C" __global__ void CUDAkrnl_697a(struct g_cu_tb __param){int t_b;float2* t_c;float2 t_d;int t_e;int a;float2* t_f;float2 t_g;int t_h;int ak;float* t_l;float dEdwi;int t_m;int an;float* t_o;float dEdli;int t_p;int aq;float* t_r;float dEdcx;int t_s;int at;float* t_u;float dEdcy;int t_v;int aw;float* t_x;float b;int t_y;int az;float* t__;float t_ab;int t_bb;int acb;float* t_db;float aeb;int t_fb;int agb;float* t_hb;float c;int akb;float4* t_lb;float4 rr;int amb;float d;float4* t_nb;float4 rrob;int apb;float dEdliqb;float dEdcxrb;float dEdcysb;float dEdwitb;int t_ub;int avb;float* t_wb;int t_xb;int ayb;float* t_zb;int t__b;int aac;float* t_bc;int t_cc;int adc;float* t_ec;t_b=blockIdx.x*blockDim.x+threadIdx.x;if(!((t_b<__param.__thread_size))){return;}t_c=(float2*)((__param.d + (t_b<<3)));t_d=*t_c;t_e=(t_b<<3);a=(__param.SPAPstride<<3);t_f=(float2*)(((__param.d + a) + t_e));t_g=*t_f;t_h=(t_b<<2);ak=(__param.SPAPstride<<4);t_l=(float*)(((__param.d + ak) + t_h));dEdwi=*t_l;t_m=(t_b<<2);an=(__param.SPAPstride*20);t_o=(float*)(((__param.d + an) + t_m));dEdli=*t_o;t_p=(t_b<<2);aq=(__param.SPAPstride*24);t_r=(float*)(((__param.d + aq) + t_p));dEdcx=*t_r;t_s=(t_b<<2);at=(__param.SPAPstride*28);t_u=(float*)(((__param.d + at) + t_s));dEdcy=*t_u;t_v=(t_b<<2);aw=(__param.SPAPstride*36);t_x=(float*)(((__param.d + aw) + t_v));b=*t_x;t_y=(t_b<<2);az=(__param.SPAPstride*40);t__=(float*)(((__param.d + az) + t_y));t_ab=*t__;t_bb=(t_b<<2);acb=(__param.SPAPstride*44);t_db=(float*)(((__param.d + acb) + t_bb));aeb=*t_db;t_fb=(t_b<<2);agb=(__param.SPAPstride*48);t_hb=(float*)(((__param.d + agb) + t_fb));c=*t_hb;akb=(__param.n+ __param.i);(__param.m_vb)[akb]=__param.ret;
	t_lb=&rr;amb=__param.i;*t_lb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,amb);;d=((dEdwi*t_ab)/ rr.x);
	t_nb=&rrob;apb=__param.i;*t_nb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,apb);;dEdliqb=(dEdli+ ((-d*b)/ rrob.x));
	dEdcxrb=(dEdcx+ (d*((-cosf((t_d).y)*cosf((t_g).y))*sinf(((t_d).x- (t_g).x)))));
	dEdcysb=(dEdcy+ (d*((cosf((t_d).y)*sinf((t_g).y))- ((sinf((t_d).y)*cosf((t_g).y))*cosf(((t_d).x- (t_g).x))))));
#line 696 "d:\\projects\\HairRenderer\\HairRenderer\\Env\\SRBFGen_src\\tool.i"
	dEdwitb=(aeb*c);
	(__param.tmp)[t_b]=dEdwitb;t_ub=(t_b<<2);avb=(__param.SPAPstride<<4);t_wb=(float*)(((__param.d + avb) + t_ub));*t_wb=dEdwitb;t_xb=(t_b<<2);ayb=(__param.SPAPstride*20);t_zb=(float*)(((__param.d + ayb) + t_xb));*t_zb=dEdliqb;t__b=(t_b<<2);aac=(__param.SPAPstride*24);t_bc=(float*)(((__param.d + aac) + t__b));*t_bc=dEdcxrb;t_cc=(t_b<<2);adc=(__param.SPAPstride*28);t_ec=(float*)(((__param.d + adc) + t_cc));*t_ec=dEdcysb;}

#line 697 "d:\\projects\\HairRenderer\\HairRenderer\\Env\\SRBFGen_src\\tool.i"
extern "C" __global__ void CUDAkrnl_697b(struct g_cu_fcgc __param){int t_c;float2* t_d;float2 t_e;int t_f;int a;float2* t_g;float2 t_h;int t_k;int al;float* t_m;float dEdwi;int t_n;int ao;float* t_p;float dEdli;int t_q;int ar;float* t_s;float dEdcx;int t_t;int au;float* t_v;float dEdcy;int t_w;int ax;float* t_y;float t_z;int t__;int aab;float* t_bb;float b;int acb;float4* t_db;float4 rr;int aeb;float d;float4* t_fb;float4 rrgb;int ahb;float dEdlikb;float dEdcxlb;float dEdcymb;t_c=blockIdx.x*blockDim.x+threadIdx.x;if(!((t_c<__param.__thread_size))){return;}t_d=(float2*)((__param.d + (t_c<<3)));t_e=*t_d;t_f=(t_c<<3);a=(__param.SPAPstride<<3);t_g=(float2*)(((__param.d + a) + t_f));t_h=*t_g;t_k=(t_c<<2);al=(__param.SPAPstride<<4);t_m=(float*)(((__param.d + al) + t_k));dEdwi=*t_m;t_n=(t_c<<2);ao=(__param.SPAPstride*20);t_p=(float*)(((__param.d + ao) + t_n));dEdli=*t_p;t_q=(t_c<<2);ar=(__param.SPAPstride*24);t_s=(float*)(((__param.d + ar) + t_q));dEdcx=*t_s;t_t=(t_c<<2);au=(__param.SPAPstride*28);t_v=(float*)(((__param.d + au) + t_t));dEdcy=*t_v;t_w=(t_c<<2);ax=(__param.SPAPstride<<5);t_y=(float*)(((__param.d + ax) + t_w));t_z=*t_y;t__=(t_c<<2);aab=(__param.SPAPstride*36);t_bb=(float*)(((__param.d + aab) + t__));b=*t_bb;acb=((__param.n<<1)+ __param.i);(__param.m_kc)[acb]=__param.ret;
	t_db=&rr;aeb=__param.i;*t_db=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,aeb);;d=((dEdwi*t_z)/ rr.x);
	t_fb=&rrgb;ahb=__param.i;*t_fb=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,ahb);;dEdlikb=(dEdli+ ((-d*b)/ rrgb.x));
	dEdcxlb=(dEdcx+ (d*((-cosf((t_e).y)*cosf((t_h).y))*sinf(((t_e).x- (t_h).x)))));
	dEdcymb=(dEdcy+ (d*((cosf((t_e).y)*sinf((t_h).y))- ((sinf((t_e).y)*cosf((t_h).y))*cosf(((t_e).x- (t_h).x))))));

	(__param.tmp)[t_c]=dEdlikb;
	(__param.m_lc)[t_c]=dEdcxlb;
	(__param.m_mc)[t_c]=dEdcymb;}

#line 703 "d:\\projects\\HairRenderer\\HairRenderer\\Env\\SRBFGen_src\\tool.i"
extern "C" __global__ void CUDAkrnl_703(struct g_cu_nb __param){int t_a;int a;int ab;int ac;t_a=blockIdx.x*blockDim.x+threadIdx.x;if(!((t_a<__param.__thread_size))){return;}a=(__param.lambda_begin+ __param.i);(__param.d)[a]=__param.ret;
	ab=(__param.centers_begin+ __param.i);(__param.d)[ab]=__param.m_ob;
	ac=((__param.centers_begin+ __param.n)+ __param.i);(__param.d)[ac]=__param.m_pb;}

#line 749 "d:\\projects\\HairRenderer\\HairRenderer\\Env\\SRBFGen_src\\tool.i"
extern "C" __global__ void CUDAkrnl_749(struct g_cu_de __param){int t_a;int b;int a;int2 t_b;float t;float bc;float td;float ae;float2 t_f;float2 t_g;float4* t_h;float4 color;float x;float texcoordx;float t_k;float tl;float xm;float texcoordy;float t_n;float to;float4 rr;float4* t_p;float4 t_q;float4 t_r;int c;float4* thiss;t_a=blockIdx.x*blockDim.x+threadIdx.x;if(!((t_a<__param.__thread_size))){return;}

	b=(t_a/ __param.width);a=(t_a%__param.width);t_b=make_int2(a,b);
	t=((float)(t_b).y/ (float)__param.height);bc=((t*1.80000000e+002f)+ -9.00000000e+001f);td=((float)(t_b).x/ (float)__param.width);ae=(td*3.60000000e+002f);t_f=make_float2(ae,bc);t_g=make_float2(((t_f).x*1.74532905e-002f),((t_f).y*1.74532905e-002f));
	t_h=&color;x=((t_g).x/ 6.28318453e+000f);if((x<1.53000012e-001f)){texcoordx=__int_as_float(0);}else{if((x>3.97000015e-001f)){t_k=1.00000000e+000f;}else{tl=((x- 1.53000012e-001f)/ 2.44000003e-001f);t_k=tl;}texcoordx=t_k;}xm=(((t_g).y+ 1.57079613e+000f)/ 3.14159226e+000f);if((xm<3.23000014e-001f)){texcoordy=__int_as_float(0);}else{if((xm>7.73000002e-001f)){t_n=1.00000000e+000f;}else{to=((xm- 3.23000014e-001f)/ 4.49999988e-001f);t_n=to;}texcoordy=t_n;}rr=tex2D(*(texture<float4,2,cudaReadModeElementType>*)&__tex0,texcoordx,texcoordy);;t_p=&rr;t_q=*t_p;t_r=make_float4((t_q).x,(t_q).y,(t_q).z,(t_q).w);*t_h=t_r;
	c=0;for(;;){if(!((c<__param.nc))){break;}
		thiss=&color;(__param.pbo_data)[((__param.nc*t_a)+ c)]=*(float*)(((char*)(thiss)+ (c<<2)));
#line 754 "d:\\projects\\HairRenderer\\HairRenderer\\Env\\SRBFGen_src\\tool.i"
		c=(c+ 1);}return;}

#line 835 "d:\\projects\\HairRenderer\\HairRenderer\\Env\\SRBFGen_src\\tool.i"
extern "C" __global__ void CUDAkrnl_835(struct g_cu_t __param){int t_a;int b;int a;int2 t_b;float t;float bc;float td;float ae;float2 t_f;float2 t_g;float4* t_h;float4 color;float x;float texcoordx;float t_k;float tl;float xm;float texcoordy;float t_n;float to;float4 rr;float4* t_p;float4 t_q;float4 t_r;int c;float4* thiss;t_a=blockIdx.x*blockDim.x+threadIdx.x;if(!((t_a<__param.__thread_size))){return;}

	b=(t_a/ __param.width);a=(t_a%__param.width);t_b=make_int2(a,b);
	t=((float)(t_b).y/ (float)__param.height);bc=((t*1.80000000e+002f)+ -9.00000000e+001f);td=((float)(t_b).x/ (float)__param.width);ae=(td*3.60000000e+002f);t_f=make_float2(ae,bc);t_g=make_float2(((t_f).x*1.74532905e-002f),((t_f).y*1.74532905e-002f));
	t_h=&color;x=((t_g).x/ 6.28318453e+000f);if((x<1.53000012e-001f)){texcoordx=__int_as_float(0);}else{if((x>3.97000015e-001f)){t_k=1.00000000e+000f;}else{tl=((x- 1.53000012e-001f)/ 2.44000003e-001f);t_k=tl;}texcoordx=t_k;}xm=(((t_g).y+ 1.57079613e+000f)/ 3.14159226e+000f);if((xm<3.23000014e-001f)){texcoordy=__int_as_float(0);}else{if((xm>7.73000002e-001f)){t_n=1.00000000e+000f;}else{to=((xm- 3.23000014e-001f)/ 4.49999988e-001f);t_n=to;}texcoordy=t_n;}rr=tex2D(*(texture<float4,2,cudaReadModeElementType>*)&__tex0,texcoordx,texcoordy);;t_p=&rr;t_q=*t_p;t_r=make_float4((t_q).x,(t_q).y,(t_q).z,(t_q).w);*t_h=t_r;
	c=0;for(;;){if(!((c<__param.nc))){break;}
		thiss=&color;(__param.pbo_data)[((__param.nc*t_a)+ c)]=*(float*)(((char*)(thiss)+ (c<<2)));
#line 840 "d:\\projects\\HairRenderer\\HairRenderer\\Env\\SRBFGen_src\\tool.i"
		c=(c+ 1);}return;}
