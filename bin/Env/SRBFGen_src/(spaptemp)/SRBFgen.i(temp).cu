extern "C" texture<int,1,cudaReadModeElementType> __tex0;
extern "C" texture<int,1,cudaReadModeElementType> __tex1;
extern "C" texture<int,1,cudaReadModeElementType> __tex2;
extern "C" texture<int,1,cudaReadModeElementType> __tex3;
struct g_cu_a;
struct pfm;
struct g_cu_c;
struct CUarray_t;
struct srbf;
struct g_cu_d;
struct g_cu_e;
struct g_cu_f;
struct g_cu_g;
struct g_cu_h;
struct g_cu_k;
struct g_cu_l;
struct g_cu_m;
struct g_cu_n;
struct g_cu_p;
struct g_cu_kb;
struct g_cu_qb;
struct g_cu_rb;
struct g_cu_ub;
struct g_cu_ac;
struct g_cu_bc;
struct g_cu_ec;
struct g_cu_hc;
struct g_cu_kc;
struct g_cu_lc;
struct g_cu_pc;
struct g_cu_tc;
struct g_cu_uc;
struct g_cu_vc;
struct g_cu_yc;
struct g_cu_ad;
struct g_cu_bd;
struct g_cu_cd;
struct g_cu_dd;
struct g_cu_ed;
struct g_cu_hd;
struct g_cu_kd;
struct g_cu_ld;
struct g_cu_md;
struct g_cu_nd;
struct g_cu_od;
struct g_cu_pd;
typedef void(*g_cu_b)(void*);
struct pfm{void* __vftab;
int __refcnt;
g_cu_b __free;
int _w;
int _h;
int _nc;
float* _raster;
struct g_cu_c* _rasterD;
struct g_cu_c* _approxD;
struct CUarray_t* _ta;
};
struct srbf{float2 _center;
float3 _weight;
float _lambda;
};
struct g_cu_a{int __thread_size;
struct pfm m_d;
char __pad[4]; struct srbf m_e;
float* d;
char __final_pad[4]; };
struct g_cu_d{float3 itembackgr;
int __thread_size;
int __isfirst;
int SPAPstride;
char* __spap_innertemp_addr;
int m_e;
float szpx;
float m_f;
float m_g;
int m_h;
int m_k;
int m_l;
int cxy;
int4* d;
float m_m;
};
struct g_cu_e{float3 itembackgr;
int __thread_size;
int __isfirst;
int SPAPstride;
char* __spap_innertemp_addr;
int m_f;
int4* d;
int m_g;
};
struct g_cu_f{int ntot;
int ofs0;
int neach;
int* scantmp;
int* a;
int* devs0;
};
typedef int g_cu_o[2312];
struct g_cu_g{int ntot;
int ofs0;
int neach;
int* scantmp;
int* devs0;
};
struct g_cu_h{int __thread_size;
int ntot;
int ofs0;
int neach;
int strt;
int* scantmp;
int* devs0;
int* a;
int* b;
};
struct g_cu_k{int __thread_size;
int ntot;
int ofs0;
int neach;
int strt;
int* scantmp;
int* devs0;
int* b;
};
struct g_cu_l{int m_m;
int n;
float* dd;
float* m_n;
};
struct g_cu_m{int __thread_size;
int* ka;
int sz;
int n;
int* kb;
int* a;
int* b;
};
struct g_cu_n{int n;
int* ka;
int __thread_size;
int* kb;
int* b;
};
struct g_cu_p{int __thread_size;
int stride;
int* d;
int* m_q;
};
struct g_cu_kb{int __thread_size;
int* m_lb;
int n;
int* m_mb;
};
struct g_cu_qb{int __thread_size;
int stride;
int* d;
int2* a;
};
struct g_cu_rb{int __thread_size;
int2* a;
int m_sb;
int* m_tb;
};
struct g_cu_ub{int __thread_size;
int* m_vb;
int n;
int2* a;
int2* b;
};
struct g_cu_ac{int __thread_size;
int2* a;
int* m_bc;
void* m_cc;
char* d;
int SPAPstride;
};
struct g_cu_bc{int __thread_size;
void* m_cc;
char* d;
int SPAPstride;
void* SPAPoldstream;
int* m_dc;
};
struct g_cu_ec{int __thread_size;
int stride;
int* d;
int* m_fc;
void* m_gc;
char* m_hc;
};
struct g_cu_hc{int __thread_size;
};
struct g_cu_kc{int __thread_size;
int* m_lc;
void* SPAPoldstream;
int* m_mc;
int* m_nc;
void* m_oc;
char* d;
};
struct g_cu_lc{int __thread_size;
void* m_mc;
char* d;
void* SPAPoldstream;
int* m_nc;
};
struct g_cu_pc{int __thread_size;
int __interrupt_base;
int __interrupt_stride;
char* d;
int m_qc;
int* m_rc;
int m_sc;
int* m_tc;
int m_uc;
int* m_vc;
int m_wc;
int* m_xc;
};
struct g_cu_tc{int __thread_size;
int* d;
int* m_uc;
int* m_vc;
int* m_wc;
int4* m_xc;
};
struct g_cu_uc{int __thread_size;
struct pfm m_vc;
int m_wc;
float m_xc;
float* m_yc;
void* m_zc;
char* d;
};
struct g_cu_vc{int __thread_size;
void* m_wc;
char* d;
float* m_xc;
};
struct g_cu_yc{int __thread_size;
};
struct g_cu_ad{int __thread_size;
struct srbf* m_bd;
void* st;
int m_cd;
};
struct g_cu_bd{int __thread_size;
int m_cd;
struct pfm m_dd;
float m_ed;
float* m_fd;
};
struct g_cu_cd{int __thread_size;
};
struct g_cu_dd{int __thread_size;
int m_ed;
struct pfm m_fd;
float m_gd;
void* st;
int m_hd;
int i;
float* m_kd;
void* m_ld;
char* d;
int SPAPstride;
};
struct g_cu_ed{int __thread_size;
void* m_fd;
char* d;
int SPAPstride;
int i;
float* m_gd;
float ret;
float* m_hd;
};
struct g_cu_hd{int __thread_size;
void* m_kd;
char* d;
int SPAPstride;
int m_ld;
int i;
float* m_md;
float ret;
float* m_nd;
};
struct g_cu_kd{int __thread_size;
void* m_ld;
char* d;
int SPAPstride;
int m_md;
int i;
float* m_nd;
float ret;
float* m_od;
float* m_pd;
float* m_qd;
};
struct g_cu_ld{int __thread_size;
int m_md;
int i;
float* d;
float ret;
int m_nd;
float m_od;
int m_pd;
float m_qd;
};
struct g_cu_md{int __thread_size;
int m_nd;
int m_od;
int m_pd;
int m_qd;
int m_rd;
int* m_sd;
};
struct g_cu_nd{int __thread_size;
int m_od;
int m_pd;
int m_qd;
float* m_rd;
};
struct g_cu_od{int __thread_size;
int a;
int c;
int m_pd;
int* m_qd;
};
struct g_cu_pd{int __thread_size;
int m_qd;
int m_rd;
int nc;
float* m_sd;
};
typedef int g_cu_qe[3];
struct g_cu_c{void* __vftab;
int __refcnt;
g_cu_b __free;
void* _d;
int _n;
int _sz;
g_cu_qe* elements;
};
struct CUarray_t{};
extern "C" __global__ void CUDAkrnl_102(struct g_cu_a __param);
extern "C" __global__ void CUDAkrnl_1172(struct g_cu_d __param);
extern "C" __global__ void CUDAkrnl_1185(struct g_cu_e __param);
extern "C" __global__ void CUDAkrnl_1473(struct g_cu_f __param);
extern "C" __global__ void CUDAkrnl_1473a(struct g_cu_g __param);
extern "C" __global__ void CUDAkrnl_1478(struct g_cu_h __param);
extern "C" __global__ void CUDAkrnl_1478a(struct g_cu_k __param);
extern "C" __global__ void CUDAkrnl_1553(struct g_cu_l __param);
extern "C" __global__ void CUDAkrnl_2250(struct g_cu_m __param);
extern "C" __global__ void CUDAkrnl_2311(struct g_cu_n __param);
extern "C" __global__ void CUDAkrnl_2768(struct g_cu_p __param);
extern "C" __global__ void CUDAkrnl_2770(struct g_cu_kb __param);
extern "C" __global__ void CUDAkrnl_2788(struct g_cu_qb __param);
extern "C" __global__ void CUDAkrnl_2792(struct g_cu_rb __param);
extern "C" __global__ void CUDAkrnl_2793(struct g_cu_ub __param);
extern "C" __global__ void CUDAkrnl_2795(struct g_cu_ac __param);
extern "C" __global__ void CUDAkrnl_2799(struct g_cu_bc __param);
extern "C" __global__ void CUDAkrnl_2805(struct g_cu_ec __param);
extern "C" __global__ void CUDAkrnl_2809(struct g_cu_hc __param);
extern "C" __global__ void CUDAkrnl_2809a(struct g_cu_kc __param);
extern "C" __global__ void CUDAkrnl_2811(struct g_cu_lc __param);
extern "C" __global__ void CUDAkrnl_2868(struct g_cu_pc __param);
extern "C" __global__ void CUDAkrnl_2871(struct g_cu_tc __param);
extern "C" __global__ void CUDAkrnl_571(struct g_cu_uc __param);
extern "C" __global__ void CUDAkrnl_587(struct g_cu_vc __param);
extern "C" __global__ void CUDAkrnl_588(struct g_cu_yc __param);
extern "C" __global__ void CUDAkrnl_593(struct g_cu_ad __param);
extern "C" __global__ void CUDAkrnl_655(struct g_cu_bd __param);
extern "C" __global__ void CUDAkrnl_670(struct g_cu_cd __param);
extern "C" __global__ void CUDAkrnl_674(struct g_cu_dd __param);
extern "C" __global__ void CUDAkrnl_697(struct g_cu_ed __param);
extern "C" __global__ void CUDAkrnl_697a(struct g_cu_hd __param);
extern "C" __global__ void CUDAkrnl_697b(struct g_cu_kd __param);
extern "C" __global__ void CUDAkrnl_703(struct g_cu_ld __param);
extern "C" __global__ void CUDAkrnl_718(struct g_cu_md __param);
extern "C" __global__ void CUDAkrnl_749(struct g_cu_nd __param);
extern "C" __global__ void CUDAkrnl_826(struct g_cu_od __param);
extern "C" __global__ void CUDAkrnl_835(struct g_cu_pd __param);

extern "C" __global__ void CUDAkrnl_102(struct g_cu_a __param){int t_f;int t_g;int t_h;float3* t_k;int2 t_l;float2 t_m;float2 t_n;float t_o;float t_p;float* t_q;float t_r;float t_s;float t_t;float t_u;float* t_v;float* t_w;float* t_x;
	t_f=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_f<__param.__thread_size)){
		t_g=(t_f%((&__param.m_d))->_w);
		t_h=(t_f/ ((&__param.m_d))->_w);
		t_k=(&((&__param.m_e))->_weight);
		t_l=make_int2(t_g,t_h);
		t_m=make_float2((((float)(t_l).x/ (float)((&__param.m_d))->_w)*3.60000000e+002f),((((float)(t_l).y/ (float)((&__param.m_d))->_h)*1.80000000e+002f)+ -9.00000000e+001f));
		t_n=make_float2(((t_m).x*1.74532905e-002f),((t_m).y*1.74532905e-002f));
		t_o=((&__param.m_e))->_lambda;
		t_p=(sinf((t_n).y)*sinf(((&__param.m_e))->_center.y));
		t_q=(float*)((&__param.m_e));
		t_r=exp((((((cosf((t_n).y)*cosf(((&__param.m_e))->_center.y))*cosf(((t_n).x- *t_q)))+ t_p)- 1.00000000e+000f)/ t_o));
		t_s=(t_r*t_k->x);
		t_t=(t_r*t_k->y);
		t_u=(t_r*t_k->z);
		t_v=(__param.d + (t_f*((&__param.m_d))->_nc));
		*t_v=(*t_v+ t_s);
		t_w=(__param.d + ((t_f*((&__param.m_d))->_nc)+ 1));
		*t_w=(*t_w+ t_t);
		t_x=(__param.d + ((t_f*((&__param.m_d))->_nc)+ 2));
		*t_x=(*t_x+ t_u);
	}
}
extern "C" __global__ void CUDAkrnl_1172(struct g_cu_d __param){int t_n;int t_o;int t_p;float t_q;float t_r;float t_s;float t_t;float t_u;float t_v;float t_w;int __loop_count;int j;int t_x;int i;float t__;float t_ab;float t_bb;float t_cb;float t_db;float t_eb;float t_fb;float t_gb;float t_hb;float t_kb;float t_mb;float t_nb;float t_ob;int t_qb;int t_rb;int4* t_sb;float t_tb;float t_ub;float t_vb;float t_wb;int t_xb;float a;float thb;float x2;float y2;int iyb;float t_ac;float t_bc;float t_cc;float t_dc;float t_ec;float t_fc;float t_gc;float t_hc;float t_kc;float t_lc;float t_mc;float t_nc;float t_oc;float t_pc;float t_qc;float t_rc;float t_sc;float t_tc;float t_uc;float t_vc;float awc;float t_xc;float t_yc;float t_zc;float t__c;float t_ad;float t_bd;float t_cd;float t_dd;float t_ed;float t_fd;float t_gd;float t_hd;float akd;float t_ld;float t_md;float t_nd;float t_od;float t_pd;float t_qd;float t_rd;float t_sd;float t_td;float t_ud;float t_vd;float t_wd;float t_xd;float t_yd;float t_zd;float* t__d;float t_ae;float t_be;float t_ce;float t_de;float t_ee;float t_fe;float t_ge;float t_he;float t_ke;float t_le;float* t_me;float t_ne;float t_oe;float t_pe;float t_qe;float t_re;float t_se;float t_te;float t_ue;float t_ve;int t_we;int t_xe;int t_ye;int t_ze;int* t__e;int* t_af;int* t_bf;int* t_cf;int* t_df;int* t_ef;int* t_ff;int* t_gf;
	t_n=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_n<__param.__thread_size)){
		if(!((__param.__isfirst==0))){
			t_o=(t_n/ __param.m_e);
			t_p=(t_n- (t_o*__param.m_e));
			t_q=(__param.m_f+ ((float)t_o*__param.szpx));
			t_r=(__param.m_g+ ((float)t_p*__param.szpx));
			t_s=__int_as_float(0);
			t_t=__int_as_float(0);
			t_u=__int_as_float(0);
			t_v=(t_r+ __param.szpx);
			t_w=(t_q+ __param.szpx);
			__loop_count=(int)t_w;
			for(j=(int)t_q;(j<=__loop_count);j+=1){
				t_x=(int)t_v;
				i=(int)t_r;
				goto g_cu_y;g_cu_z:;t__=(t_r- (float)i);
				if((__int_as_float(0)<t__)){
					t_ab=t__;
				}else{
					t_ab=__int_as_float(0);
				}
				t_bb=(t_q- (float)j);
				if((__int_as_float(0)<t_bb)){
					t_cb=t_bb;
				}else{
					t_cb=__int_as_float(0);
				}
				t_db=(t_v- (float)i);
				if((t_db<1.00000000e+000f)){
					t_eb=t_db;
				}else{
					t_eb=1.00000000e+000f;
				}
				t_fb=(t_w- (float)j);
				if((t_fb<1.00000000e+000f)){
					t_gb=t_fb;
				}else{
					t_gb=1.00000000e+000f;
				}
				if((t_ab<t_eb)){
					t_hb=t_ab;
				}else{
					t_hb=t_eb;
				}
				if((t_cb<t_gb)){
					t_kb=t_cb;
				}else{
					t_kb=t_gb;
				}
				if(!(((unsigned int)(j)>=(unsigned int)(__param.m_h)))){
					if(!(((unsigned int)(i)>=(unsigned int)(__param.m_k)))){
						if(!((((j*__param.m_k)+ i)>=__param.m_l)))goto g_cu_lb;}
				}
				t_mb=((t_eb- t_hb)*(t_gb- t_kb));
				t_nb=(t_mb*__param.itembackgr.y);
				t_ob=(t_mb*__param.itembackgr.z);
				t_s=(t_s+ (t_mb*__param.itembackgr.x));
				t_t=(t_t+ t_nb);
				t_u=(t_u+ t_ob);
				goto g_cu_pb;g_cu_lb:;t_qb=((j*__param.m_k)+ i);
				t_rb=0;
				if((t_qb==__param.cxy)){
					t_rb=1;
				}
				t_sb=(__param.d + t_qb);
				t_tb=(t_hb- 5.00000000e-001f);
				t_ub=(t_eb- 5.00000000e-001f);
				t_vb=(t_kb- 5.00000000e-001f);
				t_wb=(t_gb- 5.00000000e-001f);
				t_xb=(t_sb->w>>1);
				a=__int_as_float(0);
				thb=-1.57079613e+000f;
				if(!(((t_sb->w&1)!=0))){
					if(!(((t_xb&1)!=0))){
						thb=(-1.57079613e+000f- (3.14159226e+000f/ (float)t_xb));
					}
				}
				x2=(cosf(thb)*4.00000006e-001f);
				y2=(sinf(thb)*4.00000006e-001f);
				iyb=1;
				goto g_cu_zb;g_cu__b:;t_ac=(((((float)iyb/ (float)t_xb)*2.00000000e+000f)*3.14159226e+000f)+ thb);
				t_bc=(cosf(t_ac)*4.00000006e-001f);
				t_cc=(sinf(t_ac)*4.00000006e-001f);
				if(((t_sb->w&1)!=0)){
					t_dc=((1.00000000e+000f- (4.00000000e+000f/ (float)t_xb))*3.14159226e+000f);
					t_ec=((cosf(t_dc)- (tanf((1.57079613e+000f- (t_dc*1.50000000e+000f)))*sinf(t_dc)))*4.00000006e-001f);
					t_fc=(t_ac- (3.14159226e+000f/ (float)t_xb));
					t_gc=(t_ec*cosf(t_fc));
					t_hc=(t_ec*sinf(t_fc));
					if((x2==t_gc)){
						t_kc=__int_as_float(0);
					}else{
						t_lc=((t_hc- y2)/ (t_gc- x2));
						if((t_tb<x2)){
							t_mc=x2;
						}else{
							t_mc=t_tb;
						}
						if((t_mc<t_ub)){
							t_nc=t_mc;
						}else{
							t_nc=t_ub;
						}
						if((t_tb<t_gc)){
							t_oc=t_gc;
						}else{
							t_oc=t_tb;
						}
						if((t_oc<t_ub)){
							t_pc=t_oc;
						}else{
							t_pc=t_ub;
						}
						t_qc=(((t_nc- x2)*t_lc)+ y2);
						t_rc=(((t_pc- x2)*t_lc)+ y2);
						if((t_vb<t_qc)){
							t_sc=t_qc;
						}else{
							t_sc=t_vb;
						}
						if((t_sc<t_wb)){
							t_tc=t_sc;
						}else{
							t_tc=t_wb;
						}
						if((t_vb<t_rc)){
							t_uc=t_rc;
						}else{
							t_uc=t_vb;
						}
						if((t_uc<t_wb)){
							t_vc=t_uc;
						}else{
							t_vc=t_wb;
						}
						awc=((t_tc- t_vb)*(t_pc- t_nc));
						if((t_tc!=t_vc)){
							awc=(awc+ (((t_rc- ((t_tc+ t_vc)*5.00000000e-001f))/ t_lc)*(t_vc- t_tc)));
						}
						t_kc=awc;
					}
					a=(a+ t_kc);
					x2=t_gc;
					y2=t_hc;
				}
				if((x2==t_bc)){
					t_xc=__int_as_float(0);
				}else{
					t_yc=((t_cc- y2)/ (t_bc- x2));
					if((t_tb<x2)){
						t_zc=x2;
					}else{
						t_zc=t_tb;
					}
					if((t_zc<t_ub)){
						t__c=t_zc;
					}else{
						t__c=t_ub;
					}
					if((t_tb<t_bc)){
						t_ad=t_bc;
					}else{
						t_ad=t_tb;
					}
					if((t_ad<t_ub)){
						t_bd=t_ad;
					}else{
						t_bd=t_ub;
					}
					t_cd=(((t__c- x2)*t_yc)+ y2);
					t_dd=(((t_bd- x2)*t_yc)+ y2);
					if((t_vb<t_cd)){
						t_ed=t_cd;
					}else{
						t_ed=t_vb;
					}
					if((t_ed<t_wb)){
						t_fd=t_ed;
					}else{
						t_fd=t_wb;
					}
					if((t_vb<t_dd)){
						t_gd=t_dd;
					}else{
						t_gd=t_vb;
					}
					if((t_gd<t_wb)){
						t_hd=t_gd;
					}else{
						t_hd=t_wb;
					}
					akd=((t_fd- t_vb)*(t_bd- t__c));
					if((t_fd!=t_hd)){
						akd=(akd+ (((t_dd- ((t_fd+ t_hd)*5.00000000e-001f))/ t_yc)*(t_hd- t_fd)));
					}
					t_xc=akd;
				}
				a=(a+ t_xc);
				x2=t_bc;
				y2=t_cc;
				iyb=(iyb+ 1);
				g_cu_zb:;if((iyb<=t_xb))goto g_cu__b;if((a<__int_as_float(0))){
					t_ld=-a;
				}else{
					t_ld=a;
				}
				a=t_ld;
				t_md=((t_vb+ t_wb)*5.00000000e-001f);
				t_nd=((t_tb+ t_ub)*5.00000000e-001f);
				if((t_nd<__int_as_float(0))){
					t_od=-t_nd;
				}else{
					t_od=t_nd;
				}
				t_pd=(t_od- 4.00000006e-001f);
				if((__int_as_float(0)<t_pd)){
					t_qd=t_pd;
				}else{
					t_qd=__int_as_float(0);
				}
				t_rd=(t_qd+ 1.00000000e+000f);
				t_sd=(((sinf(((((t_md*(t_rd*t_rd))+ 5.00000000e-001f)*3.76991081e+000f)- 6.28318489e-001f))*6.00000024e-001f)+ 4.00000006e-001f)*(1.00000000e+000f- (t_qd*3.00000000e+000f)));
				t_td=__param.itembackgr.x;
				t_ud=__param.itembackgr.y;
				t_vd=__param.itembackgr.z;
				if((t_rb!=0)){
					t_wd=(t_ud*1.50000000e+000f);
					t_xd=(t_vd*1.50000000e+000f);
					t_td=(t_td*1.50000000e+000f);
					t_ud=t_wd;
					t_vd=t_xd;
				}
				t_yd=((t_sd*5.00000000e-001f)+ 5.00000000e-001f);
				t_zd=(t_sd*1.50000006e-001f);
				t__d=(float*)(t_sb);
				t_ae=*t__d;
				t_be=*(float*)((&t_sb->y));
				t_ce=*(float*)((&t_sb->z));
				if((t_ae<__int_as_float(0))){
					t_de=-t_ae;
				}else{
					t_de=t_ae;
				}
				t_ee=(1.00000000e+000f- (1.00000000e+000f/ (t_de+ 1.00000000e+000f)));
				if((t_be<__int_as_float(0))){
					t_fe=-t_be;
				}else{
					t_fe=t_be;
				}
				t_ge=(1.00000000e+000f- (1.00000000e+000f/ (t_fe+ 1.00000000e+000f)));
				if((t_ce<__int_as_float(0))){
					t_he=-t_ce;
				}else{
					t_he=t_ce;
				}
				t_ke=(t_yd*((t_zd*(t_ge- t_ud))+ t_ud));
				t_le=(t_yd*((t_zd*((1.00000000e+000f- (1.00000000e+000f/ (t_he+ 1.00000000e+000f)))- t_vd))+ t_vd));
				t_td=(t_yd*((t_zd*(t_ee- t_td))+ t_td));
				t_ud=t_ke;
				t_vd=t_le;
				t_me=(float*)(t_sb);
				t_ne=(a**t_me);
				t_oe=(a**(float*)((&t_sb->y)));
				t_pe=(a**(float*)((&t_sb->z)));
				t_qe=(((t_ub- t_tb)*(t_wb- t_vb))- a);
				t_re=((t_qe*t_ud)+ t_oe);
				t_se=((t_qe*t_vd)+ t_pe);
				t_s=(t_s+ ((t_qe*t_td)+ t_ne));
				t_t=(t_t+ t_re);
				t_u=(t_u+ t_se);
				g_cu_pb:;i=(i+ 1);
				g_cu_y:;if((i<=t_x))goto g_cu_z;}
			t_te=(1.00000000e+000f/ (__param.szpx*__param.szpx));
			t_ue=(__param.m_m*(t_te*t_t));
			t_ve=(__param.m_m*(t_te*t_u));
			t_we=__float_as_int((__param.m_m*(t_te*t_s)));
			t_xe=__float_as_int(t_ue);
			t_ye=__float_as_int(t_ve);
			t_ze=((t_o<<16)+ t_p);
			t__e=(int*)((__param.__spap_innertemp_addr + (t_n<<2)));
			*t__e=1;
			t_af=(int*)(((__param.__spap_innertemp_addr + (__param.SPAPstride<<2)) + (t_n<<2)));
			*t_af=1;
			t_bf=(int*)(((__param.__spap_innertemp_addr + (__param.SPAPstride*12)) + (t_n<<2)));
			*t_bf=t_we;
			t_cf=(int*)(((__param.__spap_innertemp_addr + (__param.SPAPstride<<4)) + (t_n<<2)));
			*t_cf=t_xe;
			t_df=(int*)(((__param.__spap_innertemp_addr + (__param.SPAPstride*20)) + (t_n<<2)));
			*t_df=t_ye;
			t_ef=(int*)(((__param.__spap_innertemp_addr + (__param.SPAPstride*24)) + (t_n<<2)));
			*t_ef=t_ze;
		}else{
			t_ff=(int*)((__param.__spap_innertemp_addr + (t_n<<2)));
			*t_ff=0;
			t_gf=(int*)(((__param.__spap_innertemp_addr + (__param.SPAPstride<<2)) + (t_n<<2)));
			*t_gf=0;
		}
	}
}
extern "C" __global__ void CUDAkrnl_1185(struct g_cu_e __param){int t_h;int t_k;int t_l;float t_m;float t_n;float t_o;float t_p;float t_q;float t_r;float t_s;int __loop_count;int j;int t_t;int i;float t_w;float t_x;float t_y;float t_z;float t__;float t_ab;float t_bb;float t_cb;float t_db;float t_eb;float t_gb;float t_hb;float t_kb;int t_nb;int t_ob;int4* t_pb;float t_qb;float t_rb;float t_sb;float t_tb;int t_ub;float a;float thb;float x2;float y2;int ivb;float t_yb;float t_zb;float t__b;float t_ac;float t_bc;float t_cc;float t_dc;float t_ec;float t_fc;float t_gc;float t_hc;float t_kc;float t_lc;float t_mc;float t_nc;float t_oc;float t_pc;float t_qc;float t_rc;float t_sc;float atc;float t_uc;float t_vc;float t_wc;float t_xc;float t_yc;float t_zc;float t__c;float t_ad;float t_bd;float t_cd;float t_dd;float t_ed;float afd;float t_gd;float t_hd;float t_kd;float t_ld;float t_md;float t_nd;float t_od;float t_pd;float t_qd;float t_rd;float t_sd;float t_td;float t_ud;float t_vd;float t_wd;float* t_xd;float t_yd;float t_zd;float t__d;float t_ae;float t_be;float t_ce;float t_de;float t_ee;float t_fe;float t_ge;float* t_he;float t_ke;float t_le;float t_me;float t_ne;float t_oe;float t_pe;float t_qe;float t_re;int t_se;int t_te;int t_ue;int t_ve;int t_we;int t_xe;int* t_ye;int* t_ze;int* t__e;int* t_af;int* t_bf;int* t_cf;int* t_df;int* t_ef;
	t_h=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_h<__param.__thread_size)){
		if(!((__param.__isfirst==0))){
			t_k=(t_h/ 20);
			t_l=(t_h- (t_k*20));
			t_m=((float)t_k/ 2.00000000e+001f);
			t_n=((float)t_l/ 2.00000000e+001f);
			t_o=__int_as_float(0);
			t_p=__int_as_float(0);
			t_q=__int_as_float(0);
			t_r=(t_n+ 5.00000007e-002f);
			t_s=(t_m+ 5.00000007e-002f);
			__loop_count=(int)t_s;
			for(j=(int)t_m;(j<=__loop_count);j+=1){
				t_t=(int)t_r;
				i=(int)t_n;
				goto g_cu_u;g_cu_v:;t_w=(t_n- (float)i);
				if((__int_as_float(0)<t_w)){
					t_x=t_w;
				}else{
					t_x=__int_as_float(0);
				}
				t_y=(t_m- (float)j);
				if((__int_as_float(0)<t_y)){
					t_z=t_y;
				}else{
					t_z=__int_as_float(0);
				}
				t__=(t_r- (float)i);
				if((t__<1.00000000e+000f)){
					t_ab=t__;
				}else{
					t_ab=1.00000000e+000f;
				}
				t_bb=(t_s- (float)j);
				if((t_bb<1.00000000e+000f)){
					t_cb=t_bb;
				}else{
					t_cb=1.00000000e+000f;
				}
				if((t_x<t_ab)){
					t_db=t_x;
				}else{
					t_db=t_ab;
				}
				if((t_z<t_cb)){
					t_eb=t_z;
				}else{
					t_eb=t_cb;
				}
				if(!(((unsigned int)(j)>=(unsigned int)(__param.m_f)))){
					if(!(((unsigned int)(i)>=(unsigned int)(1)))){
						if(!(((j+ i)>=__param.m_f)))goto g_cu_fb;}
				}
				t_gb=((t_ab- t_db)*(t_cb- t_eb));
				t_hb=(t_gb*__param.itembackgr.y);
				t_kb=(t_gb*__param.itembackgr.z);
				t_o=(t_o+ (t_gb*__param.itembackgr.x));
				t_p=(t_p+ t_hb);
				t_q=(t_q+ t_kb);
				goto g_cu_mb;g_cu_fb:;t_nb=(j+ i);
				t_ob=0;
				if((t_nb==-1)){
					t_ob=1;
				}
				t_pb=(__param.d + t_nb);
				t_qb=(t_db- 5.00000000e-001f);
				t_rb=(t_ab- 5.00000000e-001f);
				t_sb=(t_eb- 5.00000000e-001f);
				t_tb=(t_cb- 5.00000000e-001f);
				t_ub=(t_pb->w>>1);
				a=__int_as_float(0);
				thb=-1.57079613e+000f;
				if(!(((t_pb->w&1)!=0))){
					if(!(((t_ub&1)!=0))){
						thb=(-1.57079613e+000f- (3.14159226e+000f/ (float)t_ub));
					}
				}
				x2=(cosf(thb)*4.00000006e-001f);
				y2=(sinf(thb)*4.00000006e-001f);
				ivb=1;
				goto g_cu_wb;g_cu_xb:;t_yb=(((((float)ivb/ (float)t_ub)*2.00000000e+000f)*3.14159226e+000f)+ thb);
				t_zb=(cosf(t_yb)*4.00000006e-001f);
				t__b=(sinf(t_yb)*4.00000006e-001f);
				if(((t_pb->w&1)!=0)){
					t_ac=((1.00000000e+000f- (4.00000000e+000f/ (float)t_ub))*3.14159226e+000f);
					t_bc=((cosf(t_ac)- (tanf((1.57079613e+000f- (t_ac*1.50000000e+000f)))*sinf(t_ac)))*4.00000006e-001f);
					t_cc=(t_yb- (3.14159226e+000f/ (float)t_ub));
					t_dc=(t_bc*cosf(t_cc));
					t_ec=(t_bc*sinf(t_cc));
					if((x2==t_dc)){
						t_fc=__int_as_float(0);
					}else{
						t_gc=((t_ec- y2)/ (t_dc- x2));
						if((t_qb<x2)){
							t_hc=x2;
						}else{
							t_hc=t_qb;
						}
						if((t_hc<t_rb)){
							t_kc=t_hc;
						}else{
							t_kc=t_rb;
						}
						if((t_qb<t_dc)){
							t_lc=t_dc;
						}else{
							t_lc=t_qb;
						}
						if((t_lc<t_rb)){
							t_mc=t_lc;
						}else{
							t_mc=t_rb;
						}
						t_nc=(((t_kc- x2)*t_gc)+ y2);
						t_oc=(((t_mc- x2)*t_gc)+ y2);
						if((t_sb<t_nc)){
							t_pc=t_nc;
						}else{
							t_pc=t_sb;
						}
						if((t_pc<t_tb)){
							t_qc=t_pc;
						}else{
							t_qc=t_tb;
						}
						if((t_sb<t_oc)){
							t_rc=t_oc;
						}else{
							t_rc=t_sb;
						}
						if((t_rc<t_tb)){
							t_sc=t_rc;
						}else{
							t_sc=t_tb;
						}
						atc=((t_qc- t_sb)*(t_mc- t_kc));
						if((t_qc!=t_sc)){
							atc=(atc+ (((t_oc- ((t_qc+ t_sc)*5.00000000e-001f))/ t_gc)*(t_sc- t_qc)));
						}
						t_fc=atc;
					}
					a=(a+ t_fc);
					x2=t_dc;
					y2=t_ec;
				}
				if((x2==t_zb)){
					t_uc=__int_as_float(0);
				}else{
					t_vc=((t__b- y2)/ (t_zb- x2));
					if((t_qb<x2)){
						t_wc=x2;
					}else{
						t_wc=t_qb;
					}
					if((t_wc<t_rb)){
						t_xc=t_wc;
					}else{
						t_xc=t_rb;
					}
					if((t_qb<t_zb)){
						t_yc=t_zb;
					}else{
						t_yc=t_qb;
					}
					if((t_yc<t_rb)){
						t_zc=t_yc;
					}else{
						t_zc=t_rb;
					}
					t__c=(((t_xc- x2)*t_vc)+ y2);
					t_ad=(((t_zc- x2)*t_vc)+ y2);
					if((t_sb<t__c)){
						t_bd=t__c;
					}else{
						t_bd=t_sb;
					}
					if((t_bd<t_tb)){
						t_cd=t_bd;
					}else{
						t_cd=t_tb;
					}
					if((t_sb<t_ad)){
						t_dd=t_ad;
					}else{
						t_dd=t_sb;
					}
					if((t_dd<t_tb)){
						t_ed=t_dd;
					}else{
						t_ed=t_tb;
					}
					afd=((t_cd- t_sb)*(t_zc- t_xc));
					if((t_cd!=t_ed)){
						afd=(afd+ (((t_ad- ((t_cd+ t_ed)*5.00000000e-001f))/ t_vc)*(t_ed- t_cd)));
					}
					t_uc=afd;
				}
				a=(a+ t_uc);
				x2=t_zb;
				y2=t__b;
				ivb=(ivb+ 1);
				g_cu_wb:;if((ivb<=t_ub))goto g_cu_xb;if((a<__int_as_float(0))){
					t_gd=-a;
				}else{
					t_gd=a;
				}
				a=t_gd;
				t_hd=((t_sb+ t_tb)*5.00000000e-001f);
				t_kd=((t_qb+ t_rb)*5.00000000e-001f);
				if((t_kd<__int_as_float(0))){
					t_ld=-t_kd;
				}else{
					t_ld=t_kd;
				}
				t_md=(t_ld- 4.00000006e-001f);
				if((__int_as_float(0)<t_md)){
					t_nd=t_md;
				}else{
					t_nd=__int_as_float(0);
				}
				t_od=(t_nd+ 1.00000000e+000f);
				t_pd=(((sinf(((((t_hd*(t_od*t_od))+ 5.00000000e-001f)*3.76991081e+000f)- 6.28318489e-001f))*6.00000024e-001f)+ 4.00000006e-001f)*(1.00000000e+000f- (t_nd*3.00000000e+000f)));
				t_qd=__param.itembackgr.x;
				t_rd=__param.itembackgr.y;
				t_sd=__param.itembackgr.z;
				if((t_ob!=0)){
					t_td=(t_rd*1.50000000e+000f);
					t_ud=(t_sd*1.50000000e+000f);
					t_qd=(t_qd*1.50000000e+000f);
					t_rd=t_td;
					t_sd=t_ud;
				}
				t_vd=((t_pd*5.00000000e-001f)+ 5.00000000e-001f);
				t_wd=(t_pd*1.50000006e-001f);
				t_xd=(float*)(t_pb);
				t_yd=*t_xd;
				t_zd=*(float*)((&t_pb->y));
				t__d=*(float*)((&t_pb->z));
				if((t_yd<__int_as_float(0))){
					t_ae=-t_yd;
				}else{
					t_ae=t_yd;
				}
				t_be=(1.00000000e+000f- (1.00000000e+000f/ (t_ae+ 1.00000000e+000f)));
				if((t_zd<__int_as_float(0))){
					t_ce=-t_zd;
				}else{
					t_ce=t_zd;
				}
				t_de=(1.00000000e+000f- (1.00000000e+000f/ (t_ce+ 1.00000000e+000f)));
				if((t__d<__int_as_float(0))){
					t_ee=-t__d;
				}else{
					t_ee=t__d;
				}
				t_fe=(t_vd*((t_wd*(t_de- t_rd))+ t_rd));
				t_ge=(t_vd*((t_wd*((1.00000000e+000f- (1.00000000e+000f/ (t_ee+ 1.00000000e+000f)))- t_sd))+ t_sd));
				t_qd=(t_vd*((t_wd*(t_be- t_qd))+ t_qd));
				t_rd=t_fe;
				t_sd=t_ge;
				t_he=(float*)(t_pb);
				t_ke=(a**t_he);
				t_le=(a**(float*)((&t_pb->y)));
				t_me=(a**(float*)((&t_pb->z)));
				t_ne=(((t_rb- t_qb)*(t_tb- t_sb))- a);
				t_oe=((t_ne*t_rd)+ t_le);
				t_pe=((t_ne*t_sd)+ t_me);
				t_o=(t_o+ ((t_ne*t_qd)+ t_ke));
				t_p=(t_p+ t_oe);
				t_q=(t_q+ t_pe);
				g_cu_mb:;i=(i+ 1);
				g_cu_u:;if((i<=t_t))goto g_cu_v;}
			t_qe=(t_p*3.99999969e+002f);
			t_re=(t_q*3.99999969e+002f);
			t_se=((t_k+ 8)+ ((t_k/ 20)<<2));
			t_te=(t_l+ __param.m_g);
			t_ue=__float_as_int((t_o*3.99999969e+002f));
			t_ve=__float_as_int(t_qe);
			t_we=__float_as_int(t_re);
			t_xe=((t_se<<16)+ t_te);
			t_ye=(int*)((__param.__spap_innertemp_addr + (t_h<<2)));
			*t_ye=1;
			t_ze=(int*)(((__param.__spap_innertemp_addr + (__param.SPAPstride<<2)) + (t_h<<2)));
			*t_ze=1;
			t__e=(int*)(((__param.__spap_innertemp_addr + (__param.SPAPstride*12)) + (t_h<<2)));
			*t__e=t_ue;
			t_af=(int*)(((__param.__spap_innertemp_addr + (__param.SPAPstride<<4)) + (t_h<<2)));
			*t_af=t_ve;
			t_bf=(int*)(((__param.__spap_innertemp_addr + (__param.SPAPstride*20)) + (t_h<<2)));
			*t_bf=t_we;
			t_cf=(int*)(((__param.__spap_innertemp_addr + (__param.SPAPstride*24)) + (t_h<<2)));
			*t_cf=t_xe;
		}else{
			t_df=(int*)((__param.__spap_innertemp_addr + (t_h<<2)));
			*t_df=0;
			t_ef=(int*)(((__param.__spap_innertemp_addr + (__param.SPAPstride<<2)) + (t_h<<2)));
			*t_ef=0;
		}
	}
}
extern "C" __global__ void CUDAkrnl_1473(struct g_cu_f __param){int t_g;int t_h;int t_k;int t_l;int ldofs;void* t_m;int* t_n;int* t_p;int t_r;int t_s;int c;volatile int* t_t;volatile int* t_u;int t_y;int2* t_z;int2 t__;int2 t_ab;int2 t_bb;int t_cb;int t_db;int2* t_eb;int2 t_fb;int2 t_gb;int2 t_hb;int t_kb;int t_lb;int2* t_mb;int2 t_nb;int2 t_ob;int2 t_pb;int t_qb;int t_rb;int2* t_sb;int2 t_tb;int2 t_ub;int2 t_vb;int t_wb;int t_xb;int2* t_yb;int2 t_zb;int2 t__b;int2 t_ac;int t_bc;int t_cc;int2* t_dc;int2 t_ec;int2 t_fc;int2 t_gc;int t_hc;int t_kc;int2* t_lc;int2 t_mc;int2 t_nc;int2 t_oc;int t_pc;int t_qc;int2* t_rc;int2 t_sc;int2 t_tc;int2 t_uc;int t_vc;int t_wc;int2* t_xc;int2 t_yc;int2 t_zc;int2 t__c;int t_ad;int t_bd;int2* t_cd;int2 t_dd;int2 t_ed;int2 t_fd;int t_gd;int t_hd;int2* t_kd;int2 t_ld;int2 t_md;int2 t_nd;int t_od;int t_pd;int2* t_qd;int2 t_rd;int2 t_sd;int2 t_td;int t_ud;int t_vd;int2* t_wd;int2 t_xd;int2 t_yd;int2 t_zd;int t__d;int t_ae;int2* t_be;int2 t_ce;int2 t_de;int2 t_ee;int t_fe;int t_ge;int2* t_he;int2 t_ke;int2 t_le;int2 t_me;int t_ne;int t_oe;int2* t_pe;int2 t_qe;int2 t_re;int2 t_se;int t_te;int t_ue;int t_ve;int t_we;volatile int* t_xe;int t_ye;int* t_ze;int* t__e;int* t_af;int t_bf;int t_cf;int t_df;int t_ef;int t_ff;int t_gf;int t_hf;int t_kf;int t_lf;int t_mf;int t_nf;int t_of;int t_pf;int t_qf;int t_rf;int t_sf;
	__shared__ int __sharedbase[2312];
	t_g=blockIdx.x;;
	t_h=threadIdx.x;;
	t_k=((t_g<<7)+ t_h);
	t_l=(t_h>>4);
	ldofs=(t_h&15);
	t_m=(void*)((char*)(void*)__sharedbase+0);
	t_n=(&((g_cu_o*)(t_m))[0][128]);
	t_p=(t_n + ((t_h*17)+ t_l));
	if((((__param.neach*t_g)<<7)>=(__param.ntot- __param.ofs0))){
		if(!((t_h==0)))goto g_cu_q;t_r=blockIdx.x;;
		(__param.scantmp)[t_r]=0;
	}else{
		t_s=(__param.neach*t_k);
		c=0;
		*(int*)(((char*)(t_m)+ (t_h<<2)))=(t_s+ __param.ofs0);
		t_t=(t_n + ((((t_h&-16)*17)+ t_l)+ ldofs));
		t_u=(int*)(((char*)(t_m)+ (t_l<<2)));
		__syncthreads();
		goto g_cu_w;g_cu_x:;t_y=(*t_u+ ldofs);
		if((t_y<__param.ntot)){
			t_z=(int2*)((__param.a + (t_y<<1)));
			t__=*t_z;
			t_ab=t__;
		}else{
			t_bb=make_int2(0,0);
			t_ab=t_bb;
		}
		t_cb=((t_ab).x+ (t_ab).y);
		t_db=((t_u)[8]+ ldofs);
		if((t_db<__param.ntot)){
			t_eb=(int2*)((__param.a + (t_db<<1)));
			t_fb=*t_eb;
			t_gb=t_fb;
		}else{
			t_hb=make_int2(0,0);
			t_gb=t_hb;
		}
		t_kb=((t_gb).x+ (t_gb).y);
		t_lb=((t_u)[16]+ ldofs);
		if((t_lb<__param.ntot)){
			t_mb=(int2*)((__param.a + (t_lb<<1)));
			t_nb=*t_mb;
			t_ob=t_nb;
		}else{
			t_pb=make_int2(0,0);
			t_ob=t_pb;
		}
		t_qb=((t_ob).x+ (t_ob).y);
		t_rb=((t_u)[24]+ ldofs);
		if((t_rb<__param.ntot)){
			t_sb=(int2*)((__param.a + (t_rb<<1)));
			t_tb=*t_sb;
			t_ub=t_tb;
		}else{
			t_vb=make_int2(0,0);
			t_ub=t_vb;
		}
		t_wb=((t_ub).x+ (t_ub).y);
		t_xb=((t_u)[32]+ ldofs);
		if((t_xb<__param.ntot)){
			t_yb=(int2*)((__param.a + (t_xb<<1)));
			t_zb=*t_yb;
			t__b=t_zb;
		}else{
			t_ac=make_int2(0,0);
			t__b=t_ac;
		}
		t_bc=((t__b).x+ (t__b).y);
		t_cc=((t_u)[40]+ ldofs);
		if((t_cc<__param.ntot)){
			t_dc=(int2*)((__param.a + (t_cc<<1)));
			t_ec=*t_dc;
			t_fc=t_ec;
		}else{
			t_gc=make_int2(0,0);
			t_fc=t_gc;
		}
		t_hc=((t_fc).x+ (t_fc).y);
		t_kc=((t_u)[48]+ ldofs);
		if((t_kc<__param.ntot)){
			t_lc=(int2*)((__param.a + (t_kc<<1)));
			t_mc=*t_lc;
			t_nc=t_mc;
		}else{
			t_oc=make_int2(0,0);
			t_nc=t_oc;
		}
		t_pc=((t_nc).x+ (t_nc).y);
		t_qc=((t_u)[56]+ ldofs);
		if((t_qc<__param.ntot)){
			t_rc=(int2*)((__param.a + (t_qc<<1)));
			t_sc=*t_rc;
			t_tc=t_sc;
		}else{
			t_uc=make_int2(0,0);
			t_tc=t_uc;
		}
		t_vc=((t_tc).x+ (t_tc).y);
		t_wc=((t_u)[64]+ ldofs);
		if((t_wc<__param.ntot)){
			t_xc=(int2*)((__param.a + (t_wc<<1)));
			t_yc=*t_xc;
			t_zc=t_yc;
		}else{
			t__c=make_int2(0,0);
			t_zc=t__c;
		}
		t_ad=((t_zc).x+ (t_zc).y);
		t_bd=((t_u)[72]+ ldofs);
		if((t_bd<__param.ntot)){
			t_cd=(int2*)((__param.a + (t_bd<<1)));
			t_dd=*t_cd;
			t_ed=t_dd;
		}else{
			t_fd=make_int2(0,0);
			t_ed=t_fd;
		}
		t_gd=((t_ed).x+ (t_ed).y);
		t_hd=((t_u)[80]+ ldofs);
		if((t_hd<__param.ntot)){
			t_kd=(int2*)((__param.a + (t_hd<<1)));
			t_ld=*t_kd;
			t_md=t_ld;
		}else{
			t_nd=make_int2(0,0);
			t_md=t_nd;
		}
		t_od=((t_md).x+ (t_md).y);
		t_pd=((t_u)[88]+ ldofs);
		if((t_pd<__param.ntot)){
			t_qd=(int2*)((__param.a + (t_pd<<1)));
			t_rd=*t_qd;
			t_sd=t_rd;
		}else{
			t_td=make_int2(0,0);
			t_sd=t_td;
		}
		t_ud=((t_sd).x+ (t_sd).y);
		t_vd=((t_u)[96]+ ldofs);
		if((t_vd<__param.ntot)){
			t_wd=(int2*)((__param.a + (t_vd<<1)));
			t_xd=*t_wd;
			t_yd=t_xd;
		}else{
			t_zd=make_int2(0,0);
			t_yd=t_zd;
		}
		t__d=((t_yd).x+ (t_yd).y);
		t_ae=((t_u)[104]+ ldofs);
		if((t_ae<__param.ntot)){
			t_be=(int2*)((__param.a + (t_ae<<1)));
			t_ce=*t_be;
			t_de=t_ce;
		}else{
			t_ee=make_int2(0,0);
			t_de=t_ee;
		}
		t_fe=((t_de).x+ (t_de).y);
		t_ge=((t_u)[112]+ ldofs);
		if((t_ge<__param.ntot)){
			t_he=(int2*)((__param.a + (t_ge<<1)));
			t_ke=*t_he;
			t_le=t_ke;
		}else{
			t_me=make_int2(0,0);
			t_le=t_me;
		}
		t_ne=((t_le).x+ (t_le).y);
		t_oe=((t_u)[120]+ ldofs);
		if((t_oe<__param.ntot)){
			t_pe=(int2*)((__param.a + (t_oe<<1)));
			t_qe=*t_pe;
			t_re=t_qe;
		}else{
			t_se=make_int2(0,0);
			t_re=t_se;
		}
		t_te=((t_re).x+ (t_re).y);
		*t_t=t_cb;
		(t_t)[17]=t_kb;
		(t_t)[34]=t_qb;
		(t_t)[51]=t_wb;
		(t_t)[68]=t_bc;
		(t_t)[85]=t_hc;
		(t_t)[102]=t_pc;
		(t_t)[119]=t_vc;
		(t_t)[136]=t_ad;
		(t_t)[153]=t_gd;
		(t_t)[170]=t_od;
		(t_t)[187]=t_ud;
		(t_t)[204]=t__d;
		(t_t)[221]=t_fe;
		(t_t)[238]=t_ne;
		(t_t)[255]=t_te;
		c=(c+ *t_p);
		c=(c+ (t_p)[1]);
		c=(c+ (t_p)[2]);
		c=(c+ (t_p)[3]);
		c=(c+ (t_p)[4]);
		c=(c+ (t_p)[5]);
		c=(c+ (t_p)[6]);
		c=(c+ (t_p)[7]);
		c=(c+ (t_p)[8]);
		c=(c+ (t_p)[9]);
		c=(c+ (t_p)[10]);
		c=(c+ (t_p)[11]);
		c=(c+ (t_p)[12]);
		c=(c+ (t_p)[13]);
		c=(c+ (t_p)[14]);
		c=(c+ (t_p)[15]);
		ldofs=(ldofs+ 16);
		g_cu_w:;if((ldofs<__param.neach))goto g_cu_x;__syncthreads();
		t_ue=threadIdx.x;;
		t_ve=blockIdx.x;;
		t_we=((t_ve<<7)+ t_ue);
		t_xe=(int*)(t_m);
		t_ye=threadIdx.x;;
		t_ze=(int*)((t_xe + 1));
		if((t_ye==0)){
			(t_ze)[-(1)]=0;
		}
		t__e=(int*)((t_ze + t_ye));
		t_af=(int*)((t_ze + (t_ye<<1)));
		t_bf=blockDim.x;;
		(t_ze)[((t_ye>>4)+ ((t_ye&15)*(t_bf>>4)))]=c;
		t_cf=blockDim.x;;
		if((t_ye<(256- t_cf))){
			t_df=blockDim.x;;
			(t__e)[t_df]=0;
		}
		__syncthreads();
		if((t_ye<=127)){
			(t__e)[256]=(*t_af+ (t_af)[1]);
		}
		__syncthreads();
		if((t_ye<=63)){
			(t__e)[384]=((t_af)[256]+ (t_af)[257]);
		}
		__syncthreads();
		if((t_ye<=31)){
			(t__e)[448]=((t_af)[384]+ (t_af)[385]);
		}
		__syncthreads();
		if((t_ye<=15)){
			(t__e)[480]=((t_af)[448]+ (t_af)[449]);
		}
		__syncthreads();
		if((t_ye<=7)){
			(t__e)[496]=((t_af)[480]+ (t_af)[481]);
		}
		__syncthreads();
		if((t_ye<=3)){
			(t__e)[504]=((t_af)[496]+ (t_af)[497]);
		}
		__syncthreads();
		if((t_ye<=1)){
			(t__e)[508]=((t_af)[504]+ (t_af)[505]);
		}
		__syncthreads();
		if((t_ye<=0)){
			(t__e)[510]=((t_af)[508]+ (t_af)[509]);
		}
		if((t_ye<=0)){
			t_ef=(t__e)[510];
			(t_af)[509]=t_ef;
			if((t_ye<=-1)){
				(t_af)[510]=(t_ef+ (t_af)[510]);
			}
		}
		__syncthreads();
		if((t_ye<=1)){
			t_ff=(t__e)[508];
			(t_af)[505]=t_ff;
			if((t_ye<=0)){
				(t_af)[506]=(t_ff+ (t_af)[506]);
			}
		}
		__syncthreads();
		if((t_ye<=3)){
			t_gf=(t__e)[504];
			(t_af)[497]=t_gf;
			if((t_ye<=2)){
				(t_af)[498]=(t_gf+ (t_af)[498]);
			}
		}
		__syncthreads();
		if((t_ye<=7)){
			t_hf=(t__e)[496];
			(t_af)[481]=t_hf;
			if((t_ye<=6)){
				(t_af)[482]=(t_hf+ (t_af)[482]);
			}
		}
		__syncthreads();
		if((t_ye<=15)){
			t_kf=(t__e)[480];
			(t_af)[449]=t_kf;
			if((t_ye<=14)){
				(t_af)[450]=(t_kf+ (t_af)[450]);
			}
		}
		__syncthreads();
		if((t_ye<=31)){
			t_lf=(t__e)[448];
			(t_af)[385]=t_lf;
			if((t_ye<=30)){
				(t_af)[386]=(t_lf+ (t_af)[386]);
			}
		}
		__syncthreads();
		if((t_ye<=63)){
			t_mf=(t__e)[384];
			(t_af)[257]=t_mf;
			if((t_ye<=62)){
				(t_af)[258]=(t_mf+ (t_af)[258]);
			}
		}
		__syncthreads();
		if((t_ye<=127)){
			t_nf=(t__e)[256];
			(t_af)[1]=t_nf;
			if((t_ye<=126)){
				(t_af)[2]=(t_nf+ (t_af)[2]);
			}
		}
		__syncthreads();
		t_of=threadIdx.x;;
		t_pf=blockDim.x;;
		c=(t_xe)[((t_of>>4)+ ((t_of&15)*(t_pf>>4)))];
		(__param.devs0)[t_we]=c;
		t_qf=threadIdx.x;;
		if((t_qf==0)){
			t_rf=(t_u)[511];
			t_sf=blockIdx.x;;
			(__param.scantmp)[t_sf]=t_rf;
			g_cu_q:;}
	}
}
extern "C" __global__ void CUDAkrnl_1473a(struct g_cu_g __param){int t_h;int t_k;int t_l;int t_m;int ldofs;void* t_n;int* t_o;int* t_p;int t_s;int t_t;int c;volatile int* t_u;volatile int* t_v;uint4 rr;int* t_bb;int2 t_cb;int t_db;uint4 rreb;int* t_fb;int2 t_gb;int t_hb;uint4 rrkb;int* t_lb;int2 t_mb;int t_nb;uint4 rrob;int* t_pb;int2 t_qb;int t_rb;uint4 rrsb;int* t_tb;int2 t_ub;int t_vb;uint4 rrwb;int* t_xb;int2 t_yb;int t_zb;uint4 rr_b;int* t_ac;int2 t_bc;int t_cc;uint4 rrdc;int* t_ec;int2 t_fc;int t_gc;uint4 rrhc;int* t_kc;int2 t_lc;int t_mc;uint4 rrnc;int* t_oc;int2 t_pc;int t_qc;uint4 rrrc;int* t_sc;int2 t_tc;int t_uc;uint4 rrvc;int* t_wc;int2 t_xc;int t_yc;uint4 rrzc;int* t__c;int2 t_ad;int t_bd;uint4 rrcd;int* t_dd;int2 t_ed;int t_fd;uint4 rrgd;int* t_hd;int2 t_kd;int t_ld;uint4 rrmd;int* t_nd;int2 t_od;int t_pd;int t_qd;int t_rd;int t_sd;volatile int* t_td;int t_ud;int* t_vd;int* t_wd;int* t_xd;int t_yd;int t_zd;int t__d;int t_ae;int t_be;int t_ce;int t_de;int t_ee;int t_fe;int t_ge;int t_he;int t_ke;int t_le;int t_me;int t_ne;int t_oe;
	__shared__ int __sharedbase[2312];
	t_h=blockIdx.x;;
	t_k=threadIdx.x;;
	t_l=((t_h<<7)+ t_k);
	t_m=(t_k>>4);
	ldofs=(t_k&15);
	t_n=(void*)((char*)(void*)__sharedbase+0);
	t_o=(&((g_cu_o*)(t_n))[0][128]);
	t_p=(t_o + ((t_k*17)+ t_m));
	if((((__param.neach*t_h)<<7)>=(__param.ntot- __param.ofs0))){
		if(!((t_k==0)))goto g_cu_r;t_s=blockIdx.x;;
		(__param.scantmp)[t_s]=0;
	}else{
		t_t=(__param.neach*t_l);
		c=0;
		*(int*)(((char*)(t_n)+ (t_k<<2)))=(t_t+ __param.ofs0);
		t_u=(t_o + ((((t_k&-16)*17)+ t_m)+ ldofs));
		t_v=(int*)(((char*)(t_n)+ (t_m<<2)));
		__syncthreads();
		goto g_cu__;g_cu_ab:;*&rr=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,(*t_v+ ldofs));;
		t_bb=(int*)(&rr);
		t_cb=make_int2(*t_bb,*(int*)((&rr.y)));
		t_db=((t_cb).x+ (t_cb).y);
		*&rreb=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_v)[8]+ ldofs));;
		t_fb=(int*)(&rreb);
		t_gb=make_int2(*t_fb,*(int*)((&rreb.y)));
		t_hb=((t_gb).x+ (t_gb).y);
		*&rrkb=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_v)[16]+ ldofs));;
		t_lb=(int*)(&rrkb);
		t_mb=make_int2(*t_lb,*(int*)((&rrkb.y)));
		t_nb=((t_mb).x+ (t_mb).y);
		*&rrob=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_v)[24]+ ldofs));;
		t_pb=(int*)(&rrob);
		t_qb=make_int2(*t_pb,*(int*)((&rrob.y)));
		t_rb=((t_qb).x+ (t_qb).y);
		*&rrsb=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_v)[32]+ ldofs));;
		t_tb=(int*)(&rrsb);
		t_ub=make_int2(*t_tb,*(int*)((&rrsb.y)));
		t_vb=((t_ub).x+ (t_ub).y);
		*&rrwb=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_v)[40]+ ldofs));;
		t_xb=(int*)(&rrwb);
		t_yb=make_int2(*t_xb,*(int*)((&rrwb.y)));
		t_zb=((t_yb).x+ (t_yb).y);
		*&rr_b=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_v)[48]+ ldofs));;
		t_ac=(int*)(&rr_b);
		t_bc=make_int2(*t_ac,*(int*)((&rr_b.y)));
		t_cc=((t_bc).x+ (t_bc).y);
		*&rrdc=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_v)[56]+ ldofs));;
		t_ec=(int*)(&rrdc);
		t_fc=make_int2(*t_ec,*(int*)((&rrdc.y)));
		t_gc=((t_fc).x+ (t_fc).y);
		*&rrhc=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_v)[64]+ ldofs));;
		t_kc=(int*)(&rrhc);
		t_lc=make_int2(*t_kc,*(int*)((&rrhc.y)));
		t_mc=((t_lc).x+ (t_lc).y);
		*&rrnc=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_v)[72]+ ldofs));;
		t_oc=(int*)(&rrnc);
		t_pc=make_int2(*t_oc,*(int*)((&rrnc.y)));
		t_qc=((t_pc).x+ (t_pc).y);
		*&rrrc=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_v)[80]+ ldofs));;
		t_sc=(int*)(&rrrc);
		t_tc=make_int2(*t_sc,*(int*)((&rrrc.y)));
		t_uc=((t_tc).x+ (t_tc).y);
		*&rrvc=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_v)[88]+ ldofs));;
		t_wc=(int*)(&rrvc);
		t_xc=make_int2(*t_wc,*(int*)((&rrvc.y)));
		t_yc=((t_xc).x+ (t_xc).y);
		*&rrzc=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_v)[96]+ ldofs));;
		t__c=(int*)(&rrzc);
		t_ad=make_int2(*t__c,*(int*)((&rrzc.y)));
		t_bd=((t_ad).x+ (t_ad).y);
		*&rrcd=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_v)[104]+ ldofs));;
		t_dd=(int*)(&rrcd);
		t_ed=make_int2(*t_dd,*(int*)((&rrcd.y)));
		t_fd=((t_ed).x+ (t_ed).y);
		*&rrgd=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_v)[112]+ ldofs));;
		t_hd=(int*)(&rrgd);
		t_kd=make_int2(*t_hd,*(int*)((&rrgd.y)));
		t_ld=((t_kd).x+ (t_kd).y);
		*&rrmd=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_v)[120]+ ldofs));;
		t_nd=(int*)(&rrmd);
		t_od=make_int2(*t_nd,*(int*)((&rrmd.y)));
		t_pd=((t_od).x+ (t_od).y);
		*t_u=t_db;
		(t_u)[17]=t_hb;
		(t_u)[34]=t_nb;
		(t_u)[51]=t_rb;
		(t_u)[68]=t_vb;
		(t_u)[85]=t_zb;
		(t_u)[102]=t_cc;
		(t_u)[119]=t_gc;
		(t_u)[136]=t_mc;
		(t_u)[153]=t_qc;
		(t_u)[170]=t_uc;
		(t_u)[187]=t_yc;
		(t_u)[204]=t_bd;
		(t_u)[221]=t_fd;
		(t_u)[238]=t_ld;
		(t_u)[255]=t_pd;
		c=(c+ *t_p);
		c=(c+ (t_p)[1]);
		c=(c+ (t_p)[2]);
		c=(c+ (t_p)[3]);
		c=(c+ (t_p)[4]);
		c=(c+ (t_p)[5]);
		c=(c+ (t_p)[6]);
		c=(c+ (t_p)[7]);
		c=(c+ (t_p)[8]);
		c=(c+ (t_p)[9]);
		c=(c+ (t_p)[10]);
		c=(c+ (t_p)[11]);
		c=(c+ (t_p)[12]);
		c=(c+ (t_p)[13]);
		c=(c+ (t_p)[14]);
		c=(c+ (t_p)[15]);
		ldofs=(ldofs+ 16);
		g_cu__:;if((ldofs<__param.neach))goto g_cu_ab;__syncthreads();
		t_qd=threadIdx.x;;
		t_rd=blockIdx.x;;
		t_sd=((t_rd<<7)+ t_qd);
		t_td=(int*)(t_n);
		t_ud=threadIdx.x;;
		t_vd=(int*)((t_td + 1));
		if((t_ud==0)){
			(t_vd)[-(1)]=0;
		}
		t_wd=(int*)((t_vd + t_ud));
		t_xd=(int*)((t_vd + (t_ud<<1)));
		t_yd=blockDim.x;;
		(t_vd)[((t_ud>>4)+ ((t_ud&15)*(t_yd>>4)))]=c;
		t_zd=blockDim.x;;
		if((t_ud<(256- t_zd))){
			t__d=blockDim.x;;
			(t_wd)[t__d]=0;
		}
		__syncthreads();
		if((t_ud<=127)){
			(t_wd)[256]=(*t_xd+ (t_xd)[1]);
		}
		__syncthreads();
		if((t_ud<=63)){
			(t_wd)[384]=((t_xd)[256]+ (t_xd)[257]);
		}
		__syncthreads();
		if((t_ud<=31)){
			(t_wd)[448]=((t_xd)[384]+ (t_xd)[385]);
		}
		__syncthreads();
		if((t_ud<=15)){
			(t_wd)[480]=((t_xd)[448]+ (t_xd)[449]);
		}
		__syncthreads();
		if((t_ud<=7)){
			(t_wd)[496]=((t_xd)[480]+ (t_xd)[481]);
		}
		__syncthreads();
		if((t_ud<=3)){
			(t_wd)[504]=((t_xd)[496]+ (t_xd)[497]);
		}
		__syncthreads();
		if((t_ud<=1)){
			(t_wd)[508]=((t_xd)[504]+ (t_xd)[505]);
		}
		__syncthreads();
		if((t_ud<=0)){
			(t_wd)[510]=((t_xd)[508]+ (t_xd)[509]);
		}
		if((t_ud<=0)){
			t_ae=(t_wd)[510];
			(t_xd)[509]=t_ae;
			if((t_ud<=-1)){
				(t_xd)[510]=(t_ae+ (t_xd)[510]);
			}
		}
		__syncthreads();
		if((t_ud<=1)){
			t_be=(t_wd)[508];
			(t_xd)[505]=t_be;
			if((t_ud<=0)){
				(t_xd)[506]=(t_be+ (t_xd)[506]);
			}
		}
		__syncthreads();
		if((t_ud<=3)){
			t_ce=(t_wd)[504];
			(t_xd)[497]=t_ce;
			if((t_ud<=2)){
				(t_xd)[498]=(t_ce+ (t_xd)[498]);
			}
		}
		__syncthreads();
		if((t_ud<=7)){
			t_de=(t_wd)[496];
			(t_xd)[481]=t_de;
			if((t_ud<=6)){
				(t_xd)[482]=(t_de+ (t_xd)[482]);
			}
		}
		__syncthreads();
		if((t_ud<=15)){
			t_ee=(t_wd)[480];
			(t_xd)[449]=t_ee;
			if((t_ud<=14)){
				(t_xd)[450]=(t_ee+ (t_xd)[450]);
			}
		}
		__syncthreads();
		if((t_ud<=31)){
			t_fe=(t_wd)[448];
			(t_xd)[385]=t_fe;
			if((t_ud<=30)){
				(t_xd)[386]=(t_fe+ (t_xd)[386]);
			}
		}
		__syncthreads();
		if((t_ud<=63)){
			t_ge=(t_wd)[384];
			(t_xd)[257]=t_ge;
			if((t_ud<=62)){
				(t_xd)[258]=(t_ge+ (t_xd)[258]);
			}
		}
		__syncthreads();
		if((t_ud<=127)){
			t_he=(t_wd)[256];
			(t_xd)[1]=t_he;
			if((t_ud<=126)){
				(t_xd)[2]=(t_he+ (t_xd)[2]);
			}
		}
		__syncthreads();
		t_ke=threadIdx.x;;
		t_le=blockDim.x;;
		c=(t_td)[((t_ke>>4)+ ((t_ke&15)*(t_le>>4)))];
		(__param.devs0)[t_sd]=c;
		t_me=threadIdx.x;;
		if((t_me==0)){
			t_ne=(t_v)[511];
			t_oe=blockIdx.x;;
			(__param.scantmp)[t_oe]=t_ne;
			g_cu_r:;}
	}
}
extern "C" __global__ void CUDAkrnl_1478(struct g_cu_h __param){int t_k;int __loop_count;int t_l;int t_m;int t_n;int ldofs;void* t_o;int* t_p;int* t_q;int c;int i;volatile int* t_r;volatile int* t_s;int t_cb;int2* t_db;int2 t_eb;int2 t_fb;int2 t_gb;int t_hb;int t_kb;int t_lb;int2* t_mb;int2 t_nb;int2 t_ob;int2 t_pb;int t_qb;int t_rb;int t_sb;int2* t_tb;int2 t_ub;int2 t_vb;int2 t_wb;int t_xb;int t_yb;int t_zb;int2* t__b;int2 t_ac;int2 t_bc;int2 t_cc;int t_dc;int t_ec;int t_fc;int2* t_gc;int2 t_hc;int2 t_kc;int2 t_lc;int t_mc;int t_nc;int t_oc;int2* t_pc;int2 t_qc;int2 t_rc;int2 t_sc;int t_tc;int t_uc;int t_vc;int2* t_wc;int2 t_xc;int2 t_yc;int2 t_zc;int t__c;int t_ad;int t_bd;int2* t_cd;int2 t_dd;int2 t_ed;int2 t_fd;int t_gd;int t_hd;int t_kd;int2* t_ld;int2 t_md;int2 t_nd;int2 t_od;int t_pd;int t_qd;int t_rd;int2* t_sd;int2 t_td;int2 t_ud;int2 t_vd;int t_wd;int t_xd;int t_yd;int2* t_zd;int2 t__d;int2 t_ae;int2 t_be;int t_ce;int t_de;int t_ee;int2* t_fe;int2 t_ge;int2 t_he;int2 t_ke;int t_le;int t_me;int t_ne;int2* t_oe;int2 t_pe;int2 t_qe;int2 t_re;int t_se;int t_te;int t_ue;int2* t_ve;int2 t_we;int2 t_xe;int2 t_ye;int t_ze;int t__e;int t_af;int2* t_bf;int2 t_cf;int2 t_df;int2 t_ef;int t_ff;int t_gf;int t_hf;int2* t_kf;int2 t_lf;int2 t_mf;int2 t_nf;int t_of;int t_pf;int t_qf;int t_rf;int t_sf;int t_tf;int t_uf;int t_vf;int t_wf;int t_xf;int t_yf;int t_zf;int t__f;int t_ag;int t_bg;int t_cg;int t_dg;int t_eg;int t_fg;int t_gg;int t_hg;int t_kg;int t_lg;int t_mg;int t_ng;int t_og;int t_pg;int t_qg;int t_rg;int t_sg;int t_tg;int t_ug;int t_vg;int t_wg;int t_xg;int2 t_yg;int2* t_zg;int t__g;int2 t_ah;int2* t_bh;int t_ch;int2 t_dh;int2* t_eh;int t_fh;int2 t_gh;int2* t_hh;int t_kh;int2 t_lh;int2* t_mh;int t_nh;int2 t_oh;int2* t_ph;int t_qh;int2 t_rh;int2* t_sh;int t_th;int2 t_uh;int2* t_vh;int t_wh;int2 t_xh;int2* t_yh;int t_zh;int2 t__h;int2* t_ak;int t_bk;int2 t_ck;int2* t_dk;int t_ek;int2 t_fk;int2* t_gk;int t_hk;int2 t_kk;int2* t_lk;int t_mk;int2 t_nk;int2* t_ok;int t_pk;int2 t_qk;int2* t_rk;int t_sk;int2 t_tk;int2* t_uk;
	t_k=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_k<__param.__thread_size)){
		__shared__ int __sharedbase[2312];
		__loop_count=blockIdx.x;;
		t_l=threadIdx.x;;
		t_m=((__loop_count<<7)+ t_l);
		t_n=(t_l>>4);
		ldofs=(t_l&15);
		t_o=(void*)((char*)(void*)__sharedbase+0);
		t_p=(&((g_cu_o*)(t_o))[0][128]);
		t_q=(t_p + ((t_l*17)+ t_n));
		if(!((((__param.neach*__loop_count)<<7)>=(__param.ntot- __param.ofs0)))){
			*(int*)(((char*)(t_o)+ (t_l<<2)))=((__param.neach*t_m)+ __param.ofs0);
			if((t_l==0)){
				c=__param.strt;
				for(i=0;(i<__loop_count);i+=1){
					c=(c+ (__param.scantmp)[i]);
				}
				*(int*)(((char*)(t_o)+ 512))=c;
			}
			__syncthreads();
			c=((__param.devs0)[t_m]+ *(int*)(((char*)(t_o)+ 512)));
			t_r=(t_p + ((((t_l&-16)*17)+ t_n)+ ldofs));
			t_s=(int*)(((char*)(t_o)+ (t_n<<2)));
			__syncthreads();
			goto g_cu_t;g_cu_bb:;t_cb=(*t_s+ ldofs);
			if((t_cb<__param.ntot)){
				t_db=(int2*)((__param.a + (t_cb<<1)));
				t_eb=*t_db;
				t_fb=t_eb;
			}else{
				t_gb=make_int2(0,0);
				t_fb=t_gb;
			}
			t_hb=(t_fb).x;
			t_kb=((t_fb).x+ (t_fb).y);
			t_lb=((t_s)[8]+ ldofs);
			if((t_lb<__param.ntot)){
				t_mb=(int2*)((__param.a + (t_lb<<1)));
				t_nb=*t_mb;
				t_ob=t_nb;
			}else{
				t_pb=make_int2(0,0);
				t_ob=t_pb;
			}
			t_qb=(t_ob).x;
			t_rb=((t_ob).x+ (t_ob).y);
			t_sb=((t_s)[16]+ ldofs);
			if((t_sb<__param.ntot)){
				t_tb=(int2*)((__param.a + (t_sb<<1)));
				t_ub=*t_tb;
				t_vb=t_ub;
			}else{
				t_wb=make_int2(0,0);
				t_vb=t_wb;
			}
			t_xb=(t_vb).x;
			t_yb=((t_vb).x+ (t_vb).y);
			t_zb=((t_s)[24]+ ldofs);
			if((t_zb<__param.ntot)){
				t__b=(int2*)((__param.a + (t_zb<<1)));
				t_ac=*t__b;
				t_bc=t_ac;
			}else{
				t_cc=make_int2(0,0);
				t_bc=t_cc;
			}
			t_dc=(t_bc).x;
			t_ec=((t_bc).x+ (t_bc).y);
			t_fc=((t_s)[32]+ ldofs);
			if((t_fc<__param.ntot)){
				t_gc=(int2*)((__param.a + (t_fc<<1)));
				t_hc=*t_gc;
				t_kc=t_hc;
			}else{
				t_lc=make_int2(0,0);
				t_kc=t_lc;
			}
			t_mc=(t_kc).x;
			t_nc=((t_kc).x+ (t_kc).y);
			t_oc=((t_s)[40]+ ldofs);
			if((t_oc<__param.ntot)){
				t_pc=(int2*)((__param.a + (t_oc<<1)));
				t_qc=*t_pc;
				t_rc=t_qc;
			}else{
				t_sc=make_int2(0,0);
				t_rc=t_sc;
			}
			t_tc=(t_rc).x;
			t_uc=((t_rc).x+ (t_rc).y);
			t_vc=((t_s)[48]+ ldofs);
			if((t_vc<__param.ntot)){
				t_wc=(int2*)((__param.a + (t_vc<<1)));
				t_xc=*t_wc;
				t_yc=t_xc;
			}else{
				t_zc=make_int2(0,0);
				t_yc=t_zc;
			}
			t__c=(t_yc).x;
			t_ad=((t_yc).x+ (t_yc).y);
			t_bd=((t_s)[56]+ ldofs);
			if((t_bd<__param.ntot)){
				t_cd=(int2*)((__param.a + (t_bd<<1)));
				t_dd=*t_cd;
				t_ed=t_dd;
			}else{
				t_fd=make_int2(0,0);
				t_ed=t_fd;
			}
			t_gd=(t_ed).x;
			t_hd=((t_ed).x+ (t_ed).y);
			t_kd=((t_s)[64]+ ldofs);
			if((t_kd<__param.ntot)){
				t_ld=(int2*)((__param.a + (t_kd<<1)));
				t_md=*t_ld;
				t_nd=t_md;
			}else{
				t_od=make_int2(0,0);
				t_nd=t_od;
			}
			t_pd=(t_nd).x;
			t_qd=((t_nd).x+ (t_nd).y);
			t_rd=((t_s)[72]+ ldofs);
			if((t_rd<__param.ntot)){
				t_sd=(int2*)((__param.a + (t_rd<<1)));
				t_td=*t_sd;
				t_ud=t_td;
			}else{
				t_vd=make_int2(0,0);
				t_ud=t_vd;
			}
			t_wd=(t_ud).x;
			t_xd=((t_ud).x+ (t_ud).y);
			t_yd=((t_s)[80]+ ldofs);
			if((t_yd<__param.ntot)){
				t_zd=(int2*)((__param.a + (t_yd<<1)));
				t__d=*t_zd;
				t_ae=t__d;
			}else{
				t_be=make_int2(0,0);
				t_ae=t_be;
			}
			t_ce=(t_ae).x;
			t_de=((t_ae).x+ (t_ae).y);
			t_ee=((t_s)[88]+ ldofs);
			if((t_ee<__param.ntot)){
				t_fe=(int2*)((__param.a + (t_ee<<1)));
				t_ge=*t_fe;
				t_he=t_ge;
			}else{
				t_ke=make_int2(0,0);
				t_he=t_ke;
			}
			t_le=(t_he).x;
			t_me=((t_he).x+ (t_he).y);
			t_ne=((t_s)[96]+ ldofs);
			if((t_ne<__param.ntot)){
				t_oe=(int2*)((__param.a + (t_ne<<1)));
				t_pe=*t_oe;
				t_qe=t_pe;
			}else{
				t_re=make_int2(0,0);
				t_qe=t_re;
			}
			t_se=(t_qe).x;
			t_te=((t_qe).x+ (t_qe).y);
			t_ue=((t_s)[104]+ ldofs);
			if((t_ue<__param.ntot)){
				t_ve=(int2*)((__param.a + (t_ue<<1)));
				t_we=*t_ve;
				t_xe=t_we;
			}else{
				t_ye=make_int2(0,0);
				t_xe=t_ye;
			}
			t_ze=(t_xe).x;
			t__e=((t_xe).x+ (t_xe).y);
			t_af=((t_s)[112]+ ldofs);
			if((t_af<__param.ntot)){
				t_bf=(int2*)((__param.a + (t_af<<1)));
				t_cf=*t_bf;
				t_df=t_cf;
			}else{
				t_ef=make_int2(0,0);
				t_df=t_ef;
			}
			t_ff=(t_df).x;
			t_gf=((t_df).x+ (t_df).y);
			t_hf=((t_s)[120]+ ldofs);
			if((t_hf<__param.ntot)){
				t_kf=(int2*)((__param.a + (t_hf<<1)));
				t_lf=*t_kf;
				t_mf=t_lf;
			}else{
				t_nf=make_int2(0,0);
				t_mf=t_nf;
			}
			t_of=(t_mf).x;
			t_pf=((t_mf).x+ (t_mf).y);
			*t_r=t_kb;
			(t_r)[17]=t_rb;
			(t_r)[34]=t_yb;
			(t_r)[51]=t_ec;
			(t_r)[68]=t_nc;
			(t_r)[85]=t_uc;
			(t_r)[102]=t_ad;
			(t_r)[119]=t_hd;
			(t_r)[136]=t_qd;
			(t_r)[153]=t_xd;
			(t_r)[170]=t_de;
			(t_r)[187]=t_me;
			(t_r)[204]=t_te;
			(t_r)[221]=t__e;
			(t_r)[238]=t_gf;
			(t_r)[255]=t_pf;
			t_qf=*t_q;
			*t_q=c;
			c=(c+ t_qf);
			t_rf=(t_q)[1];
			(t_q)[1]=c;
			c=(c+ t_rf);
			t_sf=(t_q)[2];
			(t_q)[2]=c;
			c=(c+ t_sf);
			t_tf=(t_q)[3];
			(t_q)[3]=c;
			c=(c+ t_tf);
			t_uf=(t_q)[4];
			(t_q)[4]=c;
			c=(c+ t_uf);
			t_vf=(t_q)[5];
			(t_q)[5]=c;
			c=(c+ t_vf);
			t_wf=(t_q)[6];
			(t_q)[6]=c;
			c=(c+ t_wf);
			t_xf=(t_q)[7];
			(t_q)[7]=c;
			c=(c+ t_xf);
			t_yf=(t_q)[8];
			(t_q)[8]=c;
			c=(c+ t_yf);
			t_zf=(t_q)[9];
			(t_q)[9]=c;
			c=(c+ t_zf);
			t__f=(t_q)[10];
			(t_q)[10]=c;
			c=(c+ t__f);
			t_ag=(t_q)[11];
			(t_q)[11]=c;
			c=(c+ t_ag);
			t_bg=(t_q)[12];
			(t_q)[12]=c;
			c=(c+ t_bg);
			t_cg=(t_q)[13];
			(t_q)[13]=c;
			c=(c+ t_cg);
			t_dg=(t_q)[14];
			(t_q)[14]=c;
			c=(c+ t_dg);
			t_eg=(t_q)[15];
			(t_q)[15]=c;
			c=(c+ t_eg);
			t_fg=*t_r;
			t_gg=(t_r)[17];
			t_hg=(t_r)[34];
			t_kg=(t_r)[51];
			t_lg=(t_r)[68];
			t_mg=(t_r)[85];
			t_ng=(t_r)[102];
			t_og=(t_r)[119];
			t_pg=(t_r)[136];
			t_qg=(t_r)[153];
			t_rg=(t_r)[170];
			t_sg=(t_r)[187];
			t_tg=(t_r)[204];
			t_ug=(t_r)[221];
			t_vg=(t_r)[238];
			t_wg=(t_r)[255];
			t_xg=(*t_s+ ldofs);
			if((t_xg<__param.ntot)){
				t_yg=make_int2(t_fg,(t_hb+ t_fg));
				t_zg=(int2*)((__param.b + (t_xg<<1)));
				*t_zg=t_yg;
			}
			t__g=((t_s)[8]+ ldofs);
			if((t__g<__param.ntot)){
				t_ah=make_int2(t_gg,(t_qb+ t_gg));
				t_bh=(int2*)((__param.b + (t__g<<1)));
				*t_bh=t_ah;
			}
			t_ch=((t_s)[16]+ ldofs);
			if((t_ch<__param.ntot)){
				t_dh=make_int2(t_hg,(t_xb+ t_hg));
				t_eh=(int2*)((__param.b + (t_ch<<1)));
				*t_eh=t_dh;
			}
			t_fh=((t_s)[24]+ ldofs);
			if((t_fh<__param.ntot)){
				t_gh=make_int2(t_kg,(t_dc+ t_kg));
				t_hh=(int2*)((__param.b + (t_fh<<1)));
				*t_hh=t_gh;
			}
			t_kh=((t_s)[32]+ ldofs);
			if((t_kh<__param.ntot)){
				t_lh=make_int2(t_lg,(t_mc+ t_lg));
				t_mh=(int2*)((__param.b + (t_kh<<1)));
				*t_mh=t_lh;
			}
			t_nh=((t_s)[40]+ ldofs);
			if((t_nh<__param.ntot)){
				t_oh=make_int2(t_mg,(t_tc+ t_mg));
				t_ph=(int2*)((__param.b + (t_nh<<1)));
				*t_ph=t_oh;
			}
			t_qh=((t_s)[48]+ ldofs);
			if((t_qh<__param.ntot)){
				t_rh=make_int2(t_ng,(t__c+ t_ng));
				t_sh=(int2*)((__param.b + (t_qh<<1)));
				*t_sh=t_rh;
			}
			t_th=((t_s)[56]+ ldofs);
			if((t_th<__param.ntot)){
				t_uh=make_int2(t_og,(t_gd+ t_og));
				t_vh=(int2*)((__param.b + (t_th<<1)));
				*t_vh=t_uh;
			}
			t_wh=((t_s)[64]+ ldofs);
			if((t_wh<__param.ntot)){
				t_xh=make_int2(t_pg,(t_pd+ t_pg));
				t_yh=(int2*)((__param.b + (t_wh<<1)));
				*t_yh=t_xh;
			}
			t_zh=((t_s)[72]+ ldofs);
			if((t_zh<__param.ntot)){
				t__h=make_int2(t_qg,(t_wd+ t_qg));
				t_ak=(int2*)((__param.b + (t_zh<<1)));
				*t_ak=t__h;
			}
			t_bk=((t_s)[80]+ ldofs);
			if((t_bk<__param.ntot)){
				t_ck=make_int2(t_rg,(t_ce+ t_rg));
				t_dk=(int2*)((__param.b + (t_bk<<1)));
				*t_dk=t_ck;
			}
			t_ek=((t_s)[88]+ ldofs);
			if((t_ek<__param.ntot)){
				t_fk=make_int2(t_sg,(t_le+ t_sg));
				t_gk=(int2*)((__param.b + (t_ek<<1)));
				*t_gk=t_fk;
			}
			t_hk=((t_s)[96]+ ldofs);
			if((t_hk<__param.ntot)){
				t_kk=make_int2(t_tg,(t_se+ t_tg));
				t_lk=(int2*)((__param.b + (t_hk<<1)));
				*t_lk=t_kk;
			}
			t_mk=((t_s)[104]+ ldofs);
			if((t_mk<__param.ntot)){
				t_nk=make_int2(t_ug,(t_ze+ t_ug));
				t_ok=(int2*)((__param.b + (t_mk<<1)));
				*t_ok=t_nk;
			}
			t_pk=((t_s)[112]+ ldofs);
			if((t_pk<__param.ntot)){
				t_qk=make_int2(t_vg,(t_ff+ t_vg));
				t_rk=(int2*)((__param.b + (t_pk<<1)));
				*t_rk=t_qk;
			}
			t_sk=((t_s)[120]+ ldofs);
			if((t_sk<__param.ntot)){
				t_tk=make_int2(t_wg,(t_of+ t_wg));
				t_uk=(int2*)((__param.b + (t_sk<<1)));
				*t_uk=t_tk;
			}
			ldofs=(ldofs+ 16);
			g_cu_t:;if((ldofs<__param.neach))goto g_cu_bb;}
	}
}
extern "C" __global__ void CUDAkrnl_1478a(struct g_cu_k __param){int t_l;int __loop_count;int t_m;int t_n;int t_o;int ldofs;void* t_p;int* t_q;int* t_r;int c;int i;volatile int* t_s;volatile int* t_t;uint4 rr;int* t_eb;int2 t_fb;int t_gb;int t_hb;uint4 rrkb;int* t_lb;int2 t_mb;int t_nb;int t_ob;uint4 rrpb;int* t_qb;int2 t_rb;int t_sb;int t_tb;uint4 rrub;int* t_vb;int2 t_wb;int t_xb;int t_yb;uint4 rrzb;int* t__b;int2 t_ac;int t_bc;int t_cc;uint4 rrdc;int* t_ec;int2 t_fc;int t_gc;int t_hc;uint4 rrkc;int* t_lc;int2 t_mc;int t_nc;int t_oc;uint4 rrpc;int* t_qc;int2 t_rc;int t_sc;int t_tc;uint4 rruc;int* t_vc;int2 t_wc;int t_xc;int t_yc;uint4 rrzc;int* t__c;int2 t_ad;int t_bd;int t_cd;uint4 rrdd;int* t_ed;int2 t_fd;int t_gd;int t_hd;uint4 rrkd;int* t_ld;int2 t_md;int t_nd;int t_od;uint4 rrpd;int* t_qd;int2 t_rd;int t_sd;int t_td;uint4 rrud;int* t_vd;int2 t_wd;int t_xd;int t_yd;uint4 rrzd;int* t__d;int2 t_ae;int t_be;int t_ce;uint4 rrde;int* t_ee;int2 t_fe;int t_ge;int t_he;int t_ke;int t_le;int t_me;int t_ne;int t_oe;int t_pe;int t_qe;int t_re;int t_se;int t_te;int t_ue;int t_ve;int t_we;int t_xe;int t_ye;int t_ze;int t__e;int t_af;int t_bf;int t_cf;int t_df;int t_ef;int t_ff;int t_gf;int t_hf;int t_kf;int t_lf;int t_mf;int t_nf;int t_of;int t_pf;int t_qf;int t_rf;int2 t_sf;int2* t_tf;int t_uf;int2 t_vf;int2* t_wf;int t_xf;int2 t_yf;int2* t_zf;int t__f;int2 t_ag;int2* t_bg;int t_cg;int2 t_dg;int2* t_eg;int t_fg;int2 t_gg;int2* t_hg;int t_kg;int2 t_lg;int2* t_mg;int t_ng;int2 t_og;int2* t_pg;int t_qg;int2 t_rg;int2* t_sg;int t_tg;int2 t_ug;int2* t_vg;int t_wg;int2 t_xg;int2* t_yg;int t_zg;int2 t__g;int2* t_ah;int t_bh;int2 t_ch;int2* t_dh;int t_eh;int2 t_fh;int2* t_gh;int t_hh;int2 t_kh;int2* t_lh;int t_mh;int2 t_nh;int2* t_oh;
	t_l=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_l<__param.__thread_size)){
		__shared__ int __sharedbase[2312];
		__loop_count=blockIdx.x;;
		t_m=threadIdx.x;;
		t_n=((__loop_count<<7)+ t_m);
		t_o=(t_m>>4);
		ldofs=(t_m&15);
		t_p=(void*)((char*)(void*)__sharedbase+0);
		t_q=(&((g_cu_o*)(t_p))[0][128]);
		t_r=(t_q + ((t_m*17)+ t_o));
		if(!((((__param.neach*__loop_count)<<7)>=(__param.ntot- __param.ofs0)))){
			*(int*)(((char*)(t_p)+ (t_m<<2)))=((__param.neach*t_n)+ __param.ofs0);
			if((t_m==0)){
				c=__param.strt;
				for(i=0;(i<__loop_count);i+=1){
					c=(c+ (__param.scantmp)[i]);
				}
				*(int*)(((char*)(t_p)+ 512))=c;
			}
			__syncthreads();
			c=((__param.devs0)[t_n]+ *(int*)(((char*)(t_p)+ 512)));
			t_s=(t_q + ((((t_m&-16)*17)+ t_o)+ ldofs));
			t_t=(int*)(((char*)(t_p)+ (t_o<<2)));
			__syncthreads();
			goto g_cu_cb;g_cu_db:;*&rr=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,(*t_t+ ldofs));;
			t_eb=(int*)(&rr);
			t_fb=make_int2(*t_eb,*(int*)((&rr.y)));
			t_gb=(t_fb).x;
			t_hb=((t_fb).x+ (t_fb).y);
			*&rrkb=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_t)[8]+ ldofs));;
			t_lb=(int*)(&rrkb);
			t_mb=make_int2(*t_lb,*(int*)((&rrkb.y)));
			t_nb=(t_mb).x;
			t_ob=((t_mb).x+ (t_mb).y);
			*&rrpb=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_t)[16]+ ldofs));;
			t_qb=(int*)(&rrpb);
			t_rb=make_int2(*t_qb,*(int*)((&rrpb.y)));
			t_sb=(t_rb).x;
			t_tb=((t_rb).x+ (t_rb).y);
			*&rrub=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_t)[24]+ ldofs));;
			t_vb=(int*)(&rrub);
			t_wb=make_int2(*t_vb,*(int*)((&rrub.y)));
			t_xb=(t_wb).x;
			t_yb=((t_wb).x+ (t_wb).y);
			*&rrzb=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_t)[32]+ ldofs));;
			t__b=(int*)(&rrzb);
			t_ac=make_int2(*t__b,*(int*)((&rrzb.y)));
			t_bc=(t_ac).x;
			t_cc=((t_ac).x+ (t_ac).y);
			*&rrdc=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_t)[40]+ ldofs));;
			t_ec=(int*)(&rrdc);
			t_fc=make_int2(*t_ec,*(int*)((&rrdc.y)));
			t_gc=(t_fc).x;
			t_hc=((t_fc).x+ (t_fc).y);
			*&rrkc=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_t)[48]+ ldofs));;
			t_lc=(int*)(&rrkc);
			t_mc=make_int2(*t_lc,*(int*)((&rrkc.y)));
			t_nc=(t_mc).x;
			t_oc=((t_mc).x+ (t_mc).y);
			*&rrpc=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_t)[56]+ ldofs));;
			t_qc=(int*)(&rrpc);
			t_rc=make_int2(*t_qc,*(int*)((&rrpc.y)));
			t_sc=(t_rc).x;
			t_tc=((t_rc).x+ (t_rc).y);
			*&rruc=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_t)[64]+ ldofs));;
			t_vc=(int*)(&rruc);
			t_wc=make_int2(*t_vc,*(int*)((&rruc.y)));
			t_xc=(t_wc).x;
			t_yc=((t_wc).x+ (t_wc).y);
			*&rrzc=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_t)[72]+ ldofs));;
			t__c=(int*)(&rrzc);
			t_ad=make_int2(*t__c,*(int*)((&rrzc.y)));
			t_bd=(t_ad).x;
			t_cd=((t_ad).x+ (t_ad).y);
			*&rrdd=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_t)[80]+ ldofs));;
			t_ed=(int*)(&rrdd);
			t_fd=make_int2(*t_ed,*(int*)((&rrdd.y)));
			t_gd=(t_fd).x;
			t_hd=((t_fd).x+ (t_fd).y);
			*&rrkd=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_t)[88]+ ldofs));;
			t_ld=(int*)(&rrkd);
			t_md=make_int2(*t_ld,*(int*)((&rrkd.y)));
			t_nd=(t_md).x;
			t_od=((t_md).x+ (t_md).y);
			*&rrpd=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_t)[96]+ ldofs));;
			t_qd=(int*)(&rrpd);
			t_rd=make_int2(*t_qd,*(int*)((&rrpd.y)));
			t_sd=(t_rd).x;
			t_td=((t_rd).x+ (t_rd).y);
			*&rrud=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_t)[104]+ ldofs));;
			t_vd=(int*)(&rrud);
			t_wd=make_int2(*t_vd,*(int*)((&rrud.y)));
			t_xd=(t_wd).x;
			t_yd=((t_wd).x+ (t_wd).y);
			*&rrzd=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_t)[112]+ ldofs));;
			t__d=(int*)(&rrzd);
			t_ae=make_int2(*t__d,*(int*)((&rrzd.y)));
			t_be=(t_ae).x;
			t_ce=((t_ae).x+ (t_ae).y);
			*&rrde=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,((t_t)[120]+ ldofs));;
			t_ee=(int*)(&rrde);
			t_fe=make_int2(*t_ee,*(int*)((&rrde.y)));
			t_ge=(t_fe).x;
			t_he=((t_fe).x+ (t_fe).y);
			*t_s=t_hb;
			(t_s)[17]=t_ob;
			(t_s)[34]=t_tb;
			(t_s)[51]=t_yb;
			(t_s)[68]=t_cc;
			(t_s)[85]=t_hc;
			(t_s)[102]=t_oc;
			(t_s)[119]=t_tc;
			(t_s)[136]=t_yc;
			(t_s)[153]=t_cd;
			(t_s)[170]=t_hd;
			(t_s)[187]=t_od;
			(t_s)[204]=t_td;
			(t_s)[221]=t_yd;
			(t_s)[238]=t_ce;
			(t_s)[255]=t_he;
			t_ke=*t_r;
			*t_r=c;
			c=(c+ t_ke);
			t_le=(t_r)[1];
			(t_r)[1]=c;
			c=(c+ t_le);
			t_me=(t_r)[2];
			(t_r)[2]=c;
			c=(c+ t_me);
			t_ne=(t_r)[3];
			(t_r)[3]=c;
			c=(c+ t_ne);
			t_oe=(t_r)[4];
			(t_r)[4]=c;
			c=(c+ t_oe);
			t_pe=(t_r)[5];
			(t_r)[5]=c;
			c=(c+ t_pe);
			t_qe=(t_r)[6];
			(t_r)[6]=c;
			c=(c+ t_qe);
			t_re=(t_r)[7];
			(t_r)[7]=c;
			c=(c+ t_re);
			t_se=(t_r)[8];
			(t_r)[8]=c;
			c=(c+ t_se);
			t_te=(t_r)[9];
			(t_r)[9]=c;
			c=(c+ t_te);
			t_ue=(t_r)[10];
			(t_r)[10]=c;
			c=(c+ t_ue);
			t_ve=(t_r)[11];
			(t_r)[11]=c;
			c=(c+ t_ve);
			t_we=(t_r)[12];
			(t_r)[12]=c;
			c=(c+ t_we);
			t_xe=(t_r)[13];
			(t_r)[13]=c;
			c=(c+ t_xe);
			t_ye=(t_r)[14];
			(t_r)[14]=c;
			c=(c+ t_ye);
			t_ze=(t_r)[15];
			(t_r)[15]=c;
			c=(c+ t_ze);
			t__e=*t_s;
			t_af=(t_s)[17];
			t_bf=(t_s)[34];
			t_cf=(t_s)[51];
			t_df=(t_s)[68];
			t_ef=(t_s)[85];
			t_ff=(t_s)[102];
			t_gf=(t_s)[119];
			t_hf=(t_s)[136];
			t_kf=(t_s)[153];
			t_lf=(t_s)[170];
			t_mf=(t_s)[187];
			t_nf=(t_s)[204];
			t_of=(t_s)[221];
			t_pf=(t_s)[238];
			t_qf=(t_s)[255];
			t_rf=(*t_t+ ldofs);
			if((t_rf<__param.ntot)){
				t_sf=make_int2(t__e,(t_gb+ t__e));
				t_tf=(int2*)((__param.b + (t_rf<<1)));
				*t_tf=t_sf;
			}
			t_uf=((t_t)[8]+ ldofs);
			if((t_uf<__param.ntot)){
				t_vf=make_int2(t_af,(t_nb+ t_af));
				t_wf=(int2*)((__param.b + (t_uf<<1)));
				*t_wf=t_vf;
			}
			t_xf=((t_t)[16]+ ldofs);
			if((t_xf<__param.ntot)){
				t_yf=make_int2(t_bf,(t_sb+ t_bf));
				t_zf=(int2*)((__param.b + (t_xf<<1)));
				*t_zf=t_yf;
			}
			t__f=((t_t)[24]+ ldofs);
			if((t__f<__param.ntot)){
				t_ag=make_int2(t_cf,(t_xb+ t_cf));
				t_bg=(int2*)((__param.b + (t__f<<1)));
				*t_bg=t_ag;
			}
			t_cg=((t_t)[32]+ ldofs);
			if((t_cg<__param.ntot)){
				t_dg=make_int2(t_df,(t_bc+ t_df));
				t_eg=(int2*)((__param.b + (t_cg<<1)));
				*t_eg=t_dg;
			}
			t_fg=((t_t)[40]+ ldofs);
			if((t_fg<__param.ntot)){
				t_gg=make_int2(t_ef,(t_gc+ t_ef));
				t_hg=(int2*)((__param.b + (t_fg<<1)));
				*t_hg=t_gg;
			}
			t_kg=((t_t)[48]+ ldofs);
			if((t_kg<__param.ntot)){
				t_lg=make_int2(t_ff,(t_nc+ t_ff));
				t_mg=(int2*)((__param.b + (t_kg<<1)));
				*t_mg=t_lg;
			}
			t_ng=((t_t)[56]+ ldofs);
			if((t_ng<__param.ntot)){
				t_og=make_int2(t_gf,(t_sc+ t_gf));
				t_pg=(int2*)((__param.b + (t_ng<<1)));
				*t_pg=t_og;
			}
			t_qg=((t_t)[64]+ ldofs);
			if((t_qg<__param.ntot)){
				t_rg=make_int2(t_hf,(t_xc+ t_hf));
				t_sg=(int2*)((__param.b + (t_qg<<1)));
				*t_sg=t_rg;
			}
			t_tg=((t_t)[72]+ ldofs);
			if((t_tg<__param.ntot)){
				t_ug=make_int2(t_kf,(t_bd+ t_kf));
				t_vg=(int2*)((__param.b + (t_tg<<1)));
				*t_vg=t_ug;
			}
			t_wg=((t_t)[80]+ ldofs);
			if((t_wg<__param.ntot)){
				t_xg=make_int2(t_lf,(t_gd+ t_lf));
				t_yg=(int2*)((__param.b + (t_wg<<1)));
				*t_yg=t_xg;
			}
			t_zg=((t_t)[88]+ ldofs);
			if((t_zg<__param.ntot)){
				t__g=make_int2(t_mf,(t_nd+ t_mf));
				t_ah=(int2*)((__param.b + (t_zg<<1)));
				*t_ah=t__g;
			}
			t_bh=((t_t)[96]+ ldofs);
			if((t_bh<__param.ntot)){
				t_ch=make_int2(t_nf,(t_sd+ t_nf));
				t_dh=(int2*)((__param.b + (t_bh<<1)));
				*t_dh=t_ch;
			}
			t_eh=((t_t)[104]+ ldofs);
			if((t_eh<__param.ntot)){
				t_fh=make_int2(t_of,(t_xd+ t_of));
				t_gh=(int2*)((__param.b + (t_eh<<1)));
				*t_gh=t_fh;
			}
			t_hh=((t_t)[112]+ ldofs);
			if((t_hh<__param.ntot)){
				t_kh=make_int2(t_pf,(t_be+ t_pf));
				t_lh=(int2*)((__param.b + (t_hh<<1)));
				*t_lh=t_kh;
			}
			t_mh=((t_t)[120]+ ldofs);
			if((t_mh<__param.ntot)){
				t_nh=make_int2(t_qf,(t_ge+ t_qf));
				t_oh=(int2*)((__param.b + (t_mh<<1)));
				*t_oh=t_nh;
			}
			ldofs=(ldofs+ 16);
			g_cu_cb:;if((ldofs<__param.neach))goto g_cu_db;}
	}
}
extern "C" __global__ void CUDAkrnl_1553(struct g_cu_l __param){int t_o;float c;int t_p;int __loop_count;int i;void* t_q;int t_r;volatile float* t_s;float t_t;int t_u;
	t_o=blockIdx.x*blockDim.x+threadIdx.x;
	__shared__ int __sharedbase[256];
	c=__int_as_float(0);
	t_p=(__param.m_m<<8);
	__loop_count=(__param.n- 1);
	i=t_o;
	for(;(i<=__loop_count);){
		c=(c+ (__param.dd)[i]);
		i=(i+ t_p);
	}
	t_q=(void*)((char*)(void*)__sharedbase+0);
	t_r=threadIdx.x;;
	t_s=(float*)(((char*)(t_q)+ (t_r<<2)));
	*t_s=c;
	__syncthreads();
	if((t_r<128)){
		*t_s=(*t_s+ (t_s)[128]);
	}
	__syncthreads();
	if((t_r<64)){
		*t_s=(*t_s+ (t_s)[64]);
	}
	__syncthreads();
	if((t_r<32)){
		*t_s=(*t_s+ (t_s)[32]);
		*t_s=(*t_s+ (t_s)[16]);
		*t_s=(*t_s+ (t_s)[8]);
		*t_s=(*t_s+ (t_s)[4]);
		*t_s=(*t_s+ (t_s)[2]);
		if((t_r==0)){
			t_t=(*t_s+ (t_s)[1]);
			t_u=blockIdx.x;;
			(__param.m_n)[t_u]=t_t;
		}
	}
}
extern "C" __global__ void CUDAkrnl_2250(struct g_cu_m __param){int t_n;int t_o;int l;int l0;int r;int t_p;uint4 rr;int* t_q;int t_fb;uint4 rrgb;int* t_hb;int t_kb;
	t_n=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_n<__param.__thread_size)){
		t_o=(__param.ka)[t_n];
		l=((t_n&-__param.sz)^__param.sz);
		l0=l;
		r=(min((l+ __param.sz),__param.n)- 1);
		if(((t_n&__param.sz)!=0)){
			for(;(l<=r);){
				t_p=((l+ r)>>1);
				*&rr=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,t_p);;
				t_q=(int*)(&rr);
				if(!((t_o<*t_q))){
					l=(t_p+ 1);
				}else{
					r=(t_p- 1);
				}
			}
			goto g_cu_s;g_cu_eb:;t_fb=((l+ r)>>1);
			*&rrgb=tex1Dfetch(*(texture<uint4,1,cudaReadModeElementType>*)&__tex0,t_fb);;
			t_hb=(int*)(&rrgb);
			if((*t_hb<t_o)){
				l=(t_fb+ 1);
			}else{
				r=(t_fb- 1);
			}
		}
		if((l<=r))goto g_cu_eb;g_cu_s:;t_kb=(((t_n&~__param.sz)- l0)+ l);
		(__param.kb)[t_kb]=t_o;
		(__param.b)[t_kb]=(__param.a)[t_n];
	}
}
extern "C" __global__ void CUDAkrnl_2311(struct g_cu_n __param){int t_o;void* t_p;void* t_q;int t_r;int k;int t_s;int t_t;int t_u;int l;int r;int c;int i;int t_kb;int ilb;int t_pb;int t_qb;int t_rb;int l0;int t_ub;int t_zb;int t__b;int t_ac;int l0bc;int t_ec;int t_hc;int t_kc;int t_lc;int l0mc;int t_pc;int t_sc;int t_tc;int t_uc;int l0vc;int t_yc;int t_ad;int t_bd;int t_cd;int l0dd;int t_ed;int t_hd;int t_kd;int t_ld;int t_md;
	t_o=blockIdx.x*blockDim.x+threadIdx.x;
	__shared__ int __sharedbase[512];
	t_p=(void*)((char*)(void*)__sharedbase+0);
	t_q=(void*)((char*)(void*)__sharedbase+1024);
	t_r=threadIdx.x;;
	if((t_o<__param.n)){
		k=(__param.ka)[t_o];
		*(int*)(((char*)(t_p)+ (t_r<<2)))=k;
	}
	__syncthreads();
	t_s=blockDim.x;;
	t_t=blockIdx.x;;
	t_u=(__param.__thread_size- (t_t*t_s));
	l=(t_r&-8);
	r=(min((l+ 8),t_u)- 1);
	c=l;
	i=l;
	goto g_cu_gb;g_cu_hb:;t_kb=0;
	if(!((k<*(int*)(((char*)(t_p)+ (i<<2)))))){
		t_kb=1;
	}
	c=(c+ t_kb);
	i=(i+ 1);
	g_cu_gb:;if((i<t_r))goto g_cu_hb;ilb=t_r;
	goto g_cu_nb;g_cu_ob:;t_pb=*(int*)(((char*)(t_p)+ (ilb<<2)));
	t_qb=0;
	if((t_pb<k)){
		t_qb=1;
	}
	c=(c+ t_qb);
	ilb=(ilb+ 1);
	g_cu_nb:;if((ilb<=r))goto g_cu_ob;__syncthreads();
	if((t_r<t_u)){
		*(int*)(((char*)(t_p)+ (c<<2)))=k;
		*(int*)(((char*)(t_q)+ (c<<2)))=t_o;
	}
	__syncthreads();
	t_rb=*(int*)(((char*)(t_p)+ (t_r<<2)));
	k=t_rb;
	l=((t_r&-8)^8);
	l0=l;
	r=(min((l+ 8),t_u)- 1);
	if(((t_r&8)!=0)){
		goto g_cu_sb;g_cu_tb:;t_ub=((l+ r)>>1);
		if(!((k<*(int*)(((char*)(t_p)+ (t_ub<<2)))))){
			l=(t_ub+ 1);
		}else{
			r=(t_ub- 1);
			g_cu_sb:;}
		if((l<=r))goto g_cu_tb;goto g_cu_vb;g_cu_yb:;t_zb=((l+ r)>>1);
		if((*(int*)(((char*)(t_p)+ (t_zb<<2)))<k)){
			l=(t_zb+ 1);
		}else{
			r=(t_zb- 1);
		}
	}
	if((l<=r))goto g_cu_yb;g_cu_vb:;t__b=*(int*)(((char*)(t_q)+ (t_r<<2)));
	__syncthreads();
	c=(((t_r&-9)- l0)+ l);
	if((t_r<t_u)){
		*(int*)(((char*)(t_p)+ (c<<2)))=t_rb;
		*(int*)(((char*)(t_q)+ (c<<2)))=t__b;
	}
	__syncthreads();
	t_ac=*(int*)(((char*)(t_p)+ (t_r<<2)));
	k=t_ac;
	l=((t_r&-16)^16);
	l0bc=l;
	r=(min((l+ 16),t_u)- 1);
	if(((t_r&16)!=0)){
		goto g_cu_cc;g_cu_dc:;t_ec=((l+ r)>>1);
		if(!((k<*(int*)(((char*)(t_p)+ (t_ec<<2)))))){
			l=(t_ec+ 1);
		}else{
			r=(t_ec- 1);
			g_cu_cc:;}
		if((l<=r))goto g_cu_dc;goto g_cu_fc;g_cu_gc:;t_hc=((l+ r)>>1);
		if((*(int*)(((char*)(t_p)+ (t_hc<<2)))<k)){
			l=(t_hc+ 1);
		}else{
			r=(t_hc- 1);
		}
	}
	if((l<=r))goto g_cu_gc;g_cu_fc:;t_kc=*(int*)(((char*)(t_q)+ (t_r<<2)));
	__syncthreads();
	c=(((t_r&-17)- l0bc)+ l);
	if((t_r<t_u)){
		*(int*)(((char*)(t_p)+ (c<<2)))=t_ac;
		*(int*)(((char*)(t_q)+ (c<<2)))=t_kc;
	}
	__syncthreads();
	t_lc=*(int*)(((char*)(t_p)+ (t_r<<2)));
	k=t_lc;
	l=((t_r&-32)^32);
	l0mc=l;
	r=(min((l+ 32),t_u)- 1);
	if(((t_r&32)!=0)){
		goto g_cu_nc;g_cu_oc:;t_pc=((l+ r)>>1);
		if(!((k<*(int*)(((char*)(t_p)+ (t_pc<<2)))))){
			l=(t_pc+ 1);
		}else{
			r=(t_pc- 1);
			g_cu_nc:;}
		if((l<=r))goto g_cu_oc;goto g_cu_qc;g_cu_rc:;t_sc=((l+ r)>>1);
		if((*(int*)(((char*)(t_p)+ (t_sc<<2)))<k)){
			l=(t_sc+ 1);
		}else{
			r=(t_sc- 1);
		}
	}
	if((l<=r))goto g_cu_rc;g_cu_qc:;t_tc=*(int*)(((char*)(t_q)+ (t_r<<2)));
	__syncthreads();
	c=(((t_r&-33)- l0mc)+ l);
	if((t_r<t_u)){
		*(int*)(((char*)(t_p)+ (c<<2)))=t_lc;
		*(int*)(((char*)(t_q)+ (c<<2)))=t_tc;
	}
	__syncthreads();
	t_uc=*(int*)(((char*)(t_p)+ (t_r<<2)));
	k=t_uc;
	l=((t_r&-64)^64);
	l0vc=l;
	r=(min((l+ 64),t_u)- 1);
	if(((t_r&64)!=0)){
		goto g_cu_wc;g_cu_xc:;t_yc=((l+ r)>>1);
		if(!((k<*(int*)(((char*)(t_p)+ (t_yc<<2)))))){
			l=(t_yc+ 1);
		}else{
			r=(t_yc- 1);
			g_cu_wc:;}
		if((l<=r))goto g_cu_xc;goto g_cu_zc;g_cu__c:;t_ad=((l+ r)>>1);
		if((*(int*)(((char*)(t_p)+ (t_ad<<2)))<k)){
			l=(t_ad+ 1);
		}else{
			r=(t_ad- 1);
		}
	}
	if((l<=r))goto g_cu__c;g_cu_zc:;t_bd=*(int*)(((char*)(t_q)+ (t_r<<2)));
	__syncthreads();
	c=(((t_r&-65)- l0vc)+ l);
	if((t_r<t_u)){
		*(int*)(((char*)(t_p)+ (c<<2)))=t_uc;
		*(int*)(((char*)(t_q)+ (c<<2)))=t_bd;
	}
	__syncthreads();
	t_cd=*(int*)(((char*)(t_p)+ (t_r<<2)));
	k=t_cd;
	l=((t_r&-128)^128);
	l0dd=l;
	r=(min((l+ 128),t_u)- 1);
	if(((t_r&128)!=0)){
		for(;(l<=r);){
			t_ed=((l+ r)>>1);
			if(!((k<*(int*)(((char*)(t_p)+ (t_ed<<2)))))){
				l=(t_ed+ 1);
			}else{
				r=(t_ed- 1);
			}
		}
		goto g_cu_fd;g_cu_gd:;t_hd=((l+ r)>>1);
		if((*(int*)(((char*)(t_p)+ (t_hd<<2)))<k)){
			l=(t_hd+ 1);
		}else{
			r=(t_hd- 1);
		}
	}
	if((l<=r))goto g_cu_gd;g_cu_fd:;t_kd=*(int*)(((char*)(t_q)+ (t_r<<2)));
	__syncthreads();
	c=(((t_r&-129)- l0dd)+ l);
	if((t_r<t_u)){
		*(int*)(((char*)(t_p)+ (c<<2)))=t_cd;
		*(int*)(((char*)(t_q)+ (c<<2)))=t_kd;
	}
	__syncthreads();
	t_ld=threadIdx.x;;
	(__param.kb)[t_o]=*(int*)(((char*)(t_p)+ (t_ld<<2)));
	t_md=threadIdx.x;;
	(__param.b)[t_o]=*(int*)(((char*)(t_q)+ (t_md<<2)));
}
extern "C" __global__ void CUDAkrnl_2768(struct g_cu_p __param){int t_r;int t_s;int t_t;
	t_r=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_r<__param.__thread_size)){
		t_s=(__param.d)[(__param.stride+ t_r)];
		t_t=0;
		if(!((t_s!=0))){
			t_t=1;
		}
		(__param.m_q)[t_r]=t_t;
	}
}
extern "C" __global__ void CUDAkrnl_2770(struct g_cu_kb __param){int t_nb;int p;int x;
	t_nb=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_nb<__param.__thread_size)){
		p=(__param.m_lb)[t_nb];
		if((t_nb==(__param.__thread_size- 1))){
			x=(__param.n- p);
		}else{
			x=((__param.m_lb)[(t_nb+ 1)]- p);
		}
		if(!((x!=0))){
			p=((t_nb- p)+ __param.n);
		}
		(__param.m_mb)[p]=t_nb;
	}
}
extern "C" __global__ void CUDAkrnl_2788(struct g_cu_qb __param){int t_rb;int2 t_sb;int2* t_tb;
	t_rb=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_rb<__param.__thread_size)){
		t_sb=make_int2((__param.d)[(__param.stride+ t_rb)],t_rb);
		t_tb=(__param.a + t_rb);
		*t_tb=t_sb;
	}
}
extern "C" __global__ void CUDAkrnl_2792(struct g_cu_rb __param){int t_ub;int* t_vb;int t_wb;int t_xb;
	t_ub=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_ub<__param.__thread_size)){
		t_vb=(int*)((__param.a + t_ub));
		t_wb=(*t_vb&__param.m_sb);
		t_xb=0;
		if(!((t_wb!=0))){
			t_xb=1;
		}
		(__param.m_tb)[t_ub]=t_xb;
	}
}
extern "C" __global__ void CUDAkrnl_2793(struct g_cu_ub __param){int t_wb;int p;int x;int2* t_xb;int2 t_yb;int2* t_zb;
	t_wb=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_wb<__param.__thread_size)){
		p=(__param.m_vb)[t_wb];
		if((t_wb==(__param.__thread_size- 1))){
			x=(__param.n- p);
		}else{
			x=((__param.m_vb)[(t_wb+ 1)]- p);
		}
		t_xb=(__param.a + t_wb);
		t_yb=*t_xb;
		if(!((x!=0))){
			p=((t_wb- p)+ __param.n);
		}
		t_zb=(__param.b + p);
		*t_zb=t_yb;
	}
}
extern "C" __global__ void CUDAkrnl_2795(struct g_cu_ac __param){int t_dc;int2* t_ec;int2 t_fc;int t_gc;int* t_hc;int* t_kc;
	t_dc=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_dc<__param.__thread_size)){
		t_ec=(__param.a + t_dc);
		t_fc=*t_ec;
		t_gc=(t_fc).x;
		(__param.m_bc)[t_dc]=(t_fc).y;
		t_hc=(int*)((__param.d + (t_dc<<2)));
		*t_hc=t_gc;
		t_kc=(int*)(((__param.d + (__param.SPAPstride<<2)) + (t_dc<<2)));
		*t_kc=t_dc;
	}
}
extern "C" __global__ void CUDAkrnl_2799(struct g_cu_bc __param){int t_ec;int* t_fc;int t_gc;int* t_hc;int t_kc;int t_lc;
	t_ec=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_ec<__param.__thread_size)){
		t_fc=(int*)((__param.d + (t_ec<<2)));
		t_gc=*t_fc;
		t_hc=(int*)(((__param.d + (__param.SPAPstride<<2)) + (t_ec<<2)));
		t_kc=*t_hc;
		if((t_kc!=0)){
			t_lc=tex1Dfetch(*(texture<int,1,cudaReadModeElementType>*)&__tex0,(t_kc- 1));;
			if(!((t_lc!=t_gc)))goto g_cu_mc;}
		(__param.m_dc)[t_gc]=t_kc;
		g_cu_mc:;}
}
extern "C" __global__ void CUDAkrnl_2805(struct g_cu_ec __param){int t_kc;int* t_lc;
	t_kc=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_kc<__param.__thread_size)){
		(__param.m_fc)[t_kc]=(__param.d)[(t_kc+ __param.stride)];
		t_lc=(int*)((__param.m_hc + (t_kc<<2)));
		*t_lc=t_kc;
	}
}
extern "C" __global__ void CUDAkrnl_2809(struct g_cu_hc __param){
}
extern "C" __global__ void CUDAkrnl_2809a(struct g_cu_kc __param){int t_pc;int t_qc;int t_rc;int* t_sc;
	t_pc=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_pc<__param.__thread_size)){
		t_qc=tex1Dfetch(*(texture<int,1,cudaReadModeElementType>*)&__tex0,(__param.m_lc)[t_pc]);;
		t_rc=(__param.m_mc)[t_pc];
		(__param.m_nc)[t_pc]=t_qc;
		t_sc=(int*)((__param.d + (t_pc<<2)));
		*t_sc=t_rc;
	}
}
extern "C" __global__ void CUDAkrnl_2811(struct g_cu_lc __param){int t_oc;int* t_pc;int t_qc;int t_rc;
	t_oc=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_oc<__param.__thread_size)){
		t_pc=(int*)((__param.d + (t_oc<<2)));
		t_qc=*t_pc;
		if((t_oc!=0)){
			t_rc=tex1Dfetch(*(texture<int,1,cudaReadModeElementType>*)&__tex0,(t_oc- 1));;
			if(!((t_rc!=t_qc)))goto g_cu_sc;}
		(__param.m_nc)[t_qc]=t_oc;
		g_cu_sc:;}
}
extern "C" __global__ void CUDAkrnl_2868(struct g_cu_pc __param){int t_yc;int* t_zc;int t__c;int* t_ad;int* t_bd;int* t_cd;int* t_dd;
	t_yc=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_yc<__param.__thread_size)){
		t_zc=(int*)(((__param.d + (__param.__interrupt_stride<<3)) + ((__param.__interrupt_base+ t_yc)<<2)));
		t__c=*t_zc;
		t_ad=(int*)(((__param.d + (__param.m_qc*__param.__interrupt_stride)) + (t__c<<2)));
		(__param.m_rc)[t_yc]=*t_ad;
		t_bd=(int*)(((__param.d + (__param.m_sc*__param.__interrupt_stride)) + (t__c<<2)));
		(__param.m_tc)[t_yc]=*t_bd;
		t_cd=(int*)(((__param.d + (__param.m_uc*__param.__interrupt_stride)) + (t__c<<2)));
		(__param.m_vc)[t_yc]=*t_cd;
		t_dd=(int*)(((__param.d + (__param.m_wc*__param.__interrupt_stride)) + (t__c<<2)));
		(__param.m_xc)[t_yc]=*t_dd;
	}
}
extern "C" __global__ void CUDAkrnl_2871(struct g_cu_tc __param){int t_yc;int4 t_zc;int4* t__c;
	t_yc=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_yc<__param.__thread_size)){
		t_zc=make_int4((__param.m_wc)[t_yc],(__param.m_vc)[t_yc],(__param.m_uc)[t_yc],(__param.d)[t_yc]);
		t__c=(__param.m_xc + t_yc);
		*t__c=t_zc;
	}
}
extern "C" __global__ void CUDAkrnl_571(struct g_cu_uc __param){int t__c;int2 t_ad;float2 t_bd;float2 t_cd;int2 t_dd;float2 t_ed;float t_fd;float t_gd;float t_hd;float t_kd;float t_ld;float t_md;float4 rr;float4* t_nd;float4 t_od;float4 t_pd;float4 t_qd;float t_rd;int t_sd;float4 rrtd;float t_ud;float4 rrvd;float t_wd;float4 rrxd;float t_yd;float4 rrzd;float t__d;float4 rrae;float t_be;int t_ce;float4 rrde;float t_ee;float4 rrfe;float t_ge;float4 rrhe;float t_ke;float4 rrle;float t_me;float4 rrne;float t_oe;int t_pe;float4 rrqe;float t_re;float4 rrse;float t_te;float4 rrue;float t_ve;float4 rrwe;float t_xe;float4 rrye;float t_ze;float* t__e;
	t__c=blockIdx.x*blockDim.x+threadIdx.x;
	if((t__c<__param.__thread_size)){
		t_ad=make_int2((t__c%((&__param.m_vc))->_w),(t__c/ ((&__param.m_vc))->_w));
		t_bd=make_float2((((float)(t_ad).x/ (float)((&__param.m_vc))->_w)*3.60000000e+002f),((((float)(t_ad).y/ (float)((&__param.m_vc))->_h)*1.80000000e+002f)+ -9.00000000e+001f));
		t_cd=make_float2(((t_bd).x*1.74532905e-002f),((t_bd).y*1.74532905e-002f));
		if((__param.m_wc!=0)){
			t_dd=make_int2((t__c%((&__param.m_vc))->_w),(t__c/ ((&__param.m_vc))->_w));
			t_ed=make_float2((((float)(t_dd).x/ (float)((&__param.m_vc))->_w)*6.28318453e+000f),((((float)(t_dd).y/ (float)((&__param.m_vc))->_h)*3.14159226e+000f)+ -1.57079613e+000f));
			t_fd=((t_ed).x/ 6.28318453e+000f);
			if((t_fd<1.53000012e-001f)){
				t_gd=__int_as_float(0);
			}else{
				if((t_fd>3.97000015e-001f)){
					t_hd=1.00000000e+000f;
				}else{
					t_hd=((t_fd- 1.53000012e-001f)/ 2.44000003e-001f);
				}
				t_gd=t_hd;
			}
			t_kd=(((t_ed).y+ 1.57079613e+000f)/ 3.14159226e+000f);
			if((t_kd<3.23000014e-001f)){
				t_ld=__int_as_float(0);
			}else{
				if((t_kd>7.73000002e-001f)){
					t_md=1.00000000e+000f;
				}else{
					t_md=((t_kd- 3.23000014e-001f)/ 4.49999988e-001f);
				}
				t_ld=t_md;
			}
			*&rr=tex2D(*(texture<float4,2,cudaReadModeElementType>*)&__tex2,t_gd,t_ld);;
			t_nd=&rr;
			t_od=*t_nd;
			t_pd=make_float4((t_od).x,(t_od).y,(t_od).z,(t_od).w);
			t_qd=t_pd;
		}
		t_rd=cosf((t_cd).y);
		t_sd=(t__c*3);
		if((__param.m_wc!=0)){
			*&rrtd=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,t_sd);;
			t_ud=((t_qd).x- rrtd.x);
		}else{
			*&rrvd=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,t_sd);;
			t_wd=rrvd.x;
			*&rrxd=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,t_sd);;
			t_ud=(rrxd.x- t_wd);
		}
		t_yd=(((t_ud*t_rd)*__param.m_xc)*t_ud);
		*&rrzd=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,t_sd);;
		t__d=rrzd.x;
		*&rrae=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,t_sd);;
		t_be=(((rrae.x*t_rd)*__param.m_xc)*t__d);
		t_ce=((t__c*3)+ 1);
		if((__param.m_wc!=0)){
			*&rrde=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,t_ce);;
			t_ee=((t_qd).y- rrde.x);
		}else{
			*&rrfe=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,t_ce);;
			t_ge=rrfe.x;
			*&rrhe=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,t_ce);;
			t_ee=(rrhe.x- t_ge);
		}
		t_ke=(t_yd+ (((t_ee*t_rd)*__param.m_xc)*t_ee));
		*&rrle=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,t_ce);;
		t_me=rrle.x;
		*&rrne=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,t_ce);;
		t_oe=(t_be+ (((rrne.x*t_rd)*__param.m_xc)*t_me));
		t_pe=((t__c*3)+ 2);
		if((__param.m_wc!=0)){
			*&rrqe=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,t_pe);;
			t_re=((t_qd).z- rrqe.x);
		}else{
			*&rrse=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,t_pe);;
			t_te=rrse.x;
			*&rrue=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,t_pe);;
			t_re=(rrue.x- t_te);
		}
		t_ve=(t_ke+ (((t_re*t_rd)*__param.m_xc)*t_re));
		*&rrwe=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,t_pe);;
		t_xe=rrwe.x;
		*&rrye=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,t_pe);;
		t_ze=(t_oe+ (((rrye.x*t_rd)*__param.m_xc)*t_xe));
		(__param.m_yc)[t__c]=t_ve;
		t__e=(float*)((__param.d + (t__c<<2)));
		*t__e=t_ze;
	}
}
extern "C" __global__ void CUDAkrnl_587(struct g_cu_vc __param){int t_yc;float* t_zc;
	t_yc=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_yc<__param.__thread_size)){
		t_zc=(float*)((__param.d + (t_yc<<2)));
		(__param.m_xc)[t_yc]=*t_zc;
	}
}
extern "C" __global__ void CUDAkrnl_588(struct g_cu_yc __param){
}
extern "C" __global__ void CUDAkrnl_593(struct g_cu_ad __param){int t_dd;struct srbf* t_ed;char* t_fd;int t_gd;float t_hd;float* t_kd;float t_ld;float* t_md;float t_nd;float* t_od;float t_pd;float* t_qd;
	t_dd=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_dd<__param.__thread_size)){
		t_ed=(__param.m_bd + t_dd);
		t_fd=*(char**)((&__param.st));
		t_gd=__param.m_cd;
		memcpy((void*)((t_fd + (t_dd<<3))),(void*)(t_ed),8);
		t_hd=t_ed->_weight.x;
		t_kd=(float*)((t_fd + ((t_dd+ (t_gd<<1))<<2)));
		*t_kd=t_hd;
		t_ld=t_ed->_weight.y;
		t_md=(float*)((t_fd + ((t_dd+ (t_gd*3))<<2)));
		*t_md=t_ld;
		t_nd=t_ed->_weight.z;
		t_od=(float*)((t_fd + ((t_dd+ (t_gd<<2))<<2)));
		*t_od=t_nd;
		t_pd=t_ed->_lambda;
		t_qd=(float*)((t_fd + ((t_dd+ (t_gd*5))<<2)));
		*t_qd=t_pd;
	}
}
extern "C" __global__ void CUDAkrnl_655(struct g_cu_bd __param){int t_gd;int2 t_hd;float2 t_kd;float t_ld;float t_md;float t_nd;float t_od;float t_pd;float t_qd;float4 rr;float4* t_rd;float4 t_sd;float4 t_td;float4 t_ud;int2 t_vd;float2 t_wd;float2 t_xd;float t_yd;int t_zd;float4 rr_d;float t_ae;float4 rrbe;float t_ce;float4 rrde;float t_ee;int t_fe;float4 rrge;float t_he;float4 rrke;float t_le;float4 rrme;float t_ne;int t_oe;float4 rrpe;float t_qe;float4 rrre;float t_se;float4 rrte;
	t_gd=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_gd<__param.__thread_size)){
		if((__param.m_cd!=0)){
			t_hd=make_int2((t_gd%((&__param.m_dd))->_w),(t_gd/ ((&__param.m_dd))->_w));
			t_kd=make_float2((((float)(t_hd).x/ (float)((&__param.m_dd))->_w)*6.28318453e+000f),((((float)(t_hd).y/ (float)((&__param.m_dd))->_h)*3.14159226e+000f)+ -1.57079613e+000f));
			t_ld=((t_kd).x/ 6.28318453e+000f);
			if((t_ld<1.53000012e-001f)){
				t_md=__int_as_float(0);
			}else{
				if((t_ld>3.97000015e-001f)){
					t_nd=1.00000000e+000f;
				}else{
					t_nd=((t_ld- 1.53000012e-001f)/ 2.44000003e-001f);
				}
				t_md=t_nd;
			}
			t_od=(((t_kd).y+ 1.57079613e+000f)/ 3.14159226e+000f);
			if((t_od<3.23000014e-001f)){
				t_pd=__int_as_float(0);
			}else{
				if((t_od>7.73000002e-001f)){
					t_qd=1.00000000e+000f;
				}else{
					t_qd=((t_od- 3.23000014e-001f)/ 4.49999988e-001f);
				}
				t_pd=t_qd;
			}
			*&rr=tex2D(*(texture<float4,2,cudaReadModeElementType>*)&__tex2,t_md,t_pd);;
			t_rd=&rr;
			t_sd=*t_rd;
			t_td=make_float4((t_sd).x,(t_sd).y,(t_sd).z,(t_sd).w);
			t_ud=t_td;
		}
		t_vd=make_int2((t_gd%((&__param.m_dd))->_w),(t_gd/ ((&__param.m_dd))->_w));
		t_wd=make_float2((((float)(t_vd).x/ (float)((&__param.m_dd))->_w)*3.60000000e+002f),((((float)(t_vd).y/ (float)((&__param.m_dd))->_h)*1.80000000e+002f)+ -9.00000000e+001f));
		t_xd=make_float2(((t_wd).x*1.74532905e-002f),((t_wd).y*1.74532905e-002f));
		t_yd=cosf((t_xd).y);
		t_zd=(t_gd*3);
		if((__param.m_cd!=0)){
			*&rr_d=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,t_zd);;
			t_ae=((t_ud).x- rr_d.x);
		}else{
			*&rrbe=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,t_zd);;
			t_ce=rrbe.x;
			*&rrde=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,t_zd);;
			t_ae=(rrde.x- t_ce);
		}
		t_ee=(((t_ae*t_yd)*t_ae)*__param.m_ed);
		t_fe=((t_gd*3)+ 1);
		if((__param.m_cd!=0)){
			*&rrge=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,t_fe);;
			t_he=((t_ud).y- rrge.x);
		}else{
			*&rrke=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,t_fe);;
			t_le=rrke.x;
			*&rrme=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,t_fe);;
			t_he=(rrme.x- t_le);
		}
		t_ne=(t_ee+ (((t_he*t_yd)*t_he)*__param.m_ed));
		t_oe=((t_gd*3)+ 2);
		if((__param.m_cd!=0)){
			*&rrpe=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,t_oe);;
			t_qe=((t_ud).z- rrpe.x);
		}else{
			*&rrre=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,t_oe);;
			t_se=rrre.x;
			*&rrte=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,t_oe);;
			t_qe=(rrte.x- t_se);
		}
		(__param.m_fd)[t_gd]=(t_ne+ (((t_qe*t_yd)*t_qe)*__param.m_ed));
	}
}
extern "C" __global__ void CUDAkrnl_670(struct g_cu_cd __param){
}
extern "C" __global__ void CUDAkrnl_674(struct g_cu_dd __param){int t_md;int2 t_nd;float2 t_od;float t_pd;float t_qd;float t_rd;float t_sd;float t_td;float t_ud;float4 rr;float4* t_vd;float4 t_wd;float4 t_xd;float4 t_yd;int2 t_zd;float2 t__d;float2 t_ae;float t_be;int t_ce;float4 rrde;float t_ee;float4 rrfe;float t_ge;float4 rrhe;float t_ke;int t_le;float4 rrme;float t_ne;float4 rroe;float t_pe;float4 rrqe;float t_re;int t_se;float4 rrte;float t_ue;float4 rrve;float t_we;float4 rrxe;float t_ye;float2* t_ze;float2 t__e;char* t_af;int t_bf;float* t_cf;float t_df;float* t_ef;float t_ff;float* t_gf;float t_hf;float t_kf;float4 rrlf;float t_mf;float t_nf;float2* t_of;float2* t_pf;float* t_qf;float* t_rf;float* t_sf;float* t_tf;float* t_uf;float* t_vf;float* t_wf;float* t_xf;
	t_md=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_md<__param.__thread_size)){
		if((__param.m_ed!=0)){
			t_nd=make_int2((t_md%((&__param.m_fd))->_w),(t_md/ ((&__param.m_fd))->_w));
			t_od=make_float2((((float)(t_nd).x/ (float)((&__param.m_fd))->_w)*6.28318453e+000f),((((float)(t_nd).y/ (float)((&__param.m_fd))->_h)*3.14159226e+000f)+ -1.57079613e+000f));
			t_pd=((t_od).x/ 6.28318453e+000f);
			if((t_pd<1.53000012e-001f)){
				t_qd=__int_as_float(0);
			}else{
				if((t_pd>3.97000015e-001f)){
					t_rd=1.00000000e+000f;
				}else{
					t_rd=((t_pd- 1.53000012e-001f)/ 2.44000003e-001f);
				}
				t_qd=t_rd;
			}
			t_sd=(((t_od).y+ 1.57079613e+000f)/ 3.14159226e+000f);
			if((t_sd<3.23000014e-001f)){
				t_td=__int_as_float(0);
			}else{
				if((t_sd>7.73000002e-001f)){
					t_ud=1.00000000e+000f;
				}else{
					t_ud=((t_sd- 3.23000014e-001f)/ 4.49999988e-001f);
				}
				t_td=t_ud;
			}
			*&rr=tex2D(*(texture<float4,2,cudaReadModeElementType>*)&__tex3,t_qd,t_td);;
			t_vd=&rr;
			t_wd=*t_vd;
			t_xd=make_float4((t_wd).x,(t_wd).y,(t_wd).z,(t_wd).w);
			t_yd=t_xd;
		}
		t_zd=make_int2((t_md%((&__param.m_fd))->_w),(t_md/ ((&__param.m_fd))->_w));
		t__d=make_float2((((float)(t_zd).x/ (float)((&__param.m_fd))->_w)*3.60000000e+002f),((((float)(t_zd).y/ (float)((&__param.m_fd))->_h)*1.80000000e+002f)+ -9.00000000e+001f));
		t_ae=make_float2(((t__d).x*1.74532905e-002f),((t__d).y*1.74532905e-002f));
		t_be=cosf((t_ae).y);
		t_ce=(t_md*3);
		if((__param.m_ed!=0)){
			*&rrde=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,t_ce);;
			t_ee=((t_yd).x- rrde.x);
		}else{
			*&rrfe=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,t_ce);;
			t_ge=rrfe.x;
			*&rrhe=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,t_ce);;
			t_ee=(rrhe.x- t_ge);
		}
		t_ke=(t_ee*((-t_be*__param.m_gd)*2.00000000e+000f));
		t_le=((t_md*3)+ 1);
		if((__param.m_ed!=0)){
			*&rrme=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,t_le);;
			t_ne=((t_yd).y- rrme.x);
		}else{
			*&rroe=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,t_le);;
			t_pe=rroe.x;
			*&rrqe=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,t_le);;
			t_ne=(rrqe.x- t_pe);
		}
		t_re=(t_ne*((-t_be*__param.m_gd)*2.00000000e+000f));
		t_se=((t_md*3)+ 2);
		if((__param.m_ed!=0)){
			*&rrte=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,t_se);;
			t_ue=((t_yd).z- rrte.x);
		}else{
			*&rrve=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,t_se);;
			t_we=rrve.x;
			*&rrxe=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex1,t_se);;
			t_ue=(rrxe.x- t_we);
		}
		t_ye=(t_ue*((-t_be*__param.m_gd)*2.00000000e+000f));
		t_ze=(float2*)((*(char**)((&__param.st)) + (__param.i<<3)));
		t__e=*t_ze;
		t_af=*(char**)((&__param.st));
		t_bf=__param.m_hd;
		t_cf=(float*)((t_af + ((__param.i+ (t_bf<<1))<<2)));
		t_df=*t_cf;
		t_ef=(float*)((t_af + ((__param.i+ (t_bf*3))<<2)));
		t_ff=*t_ef;
		t_gf=(float*)((t_af + ((__param.i+ (t_bf<<2))<<2)));
		t_hf=*t_gf;
		t_kf=((((cosf((t_ae).y)*cosf((t__e).y))*cosf(((t_ae).x- (t__e).x)))+ (sinf((t_ae).y)*sinf((t__e).y)))- 1.00000000e+000f);
		*&rrlf=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex2,__param.i);;
		t_mf=exp((t_kf/ rrlf.x));
		t_nf=(t_ke*t_mf);
		(__param.m_kd)[t_md]=t_nf;
		t_of=(float2*)((__param.d + (t_md<<3)));
		*t_of=t__e;
		t_pf=(float2*)(((__param.d + (__param.SPAPstride<<3)) + (t_md<<3)));
		*t_pf=t_ae;
		t_qf=(float*)(((__param.d + (__param.SPAPstride<<4)) + (t_md<<2)));
		*t_qf=t_df;
		t_rf=(float*)(((__param.d + (__param.SPAPstride*20)) + (t_md<<2)));
		*t_rf=t_nf;
		t_sf=(float*)(((__param.d + (__param.SPAPstride*24)) + (t_md<<2)));
		*t_sf=t_re;
		t_tf=(float*)(((__param.d + (__param.SPAPstride<<5)) + (t_md<<2)));
		*t_tf=t_hf;
		t_uf=(float*)(((__param.d + (__param.SPAPstride*36)) + (t_md<<2)));
		*t_uf=t_kf;
		t_vf=(float*)(((__param.d + (__param.SPAPstride*40)) + (t_md<<2)));
		*t_vf=t_ff;
		t_wf=(float*)(((__param.d + (__param.SPAPstride*44)) + (t_md<<2)));
		*t_wf=t_ye;
		t_xf=(float*)(((__param.d + (__param.SPAPstride*48)) + (t_md<<2)));
		*t_xf=t_mf;
	}
}
extern "C" __global__ void CUDAkrnl_697(struct g_cu_ed __param){int t_kd;float2* t_ld;float2 t_md;float2* t_nd;float2 t_od;float* t_pd;float t_qd;float* t_rd;float t_sd;float* t_td;float t_ud;float* t_vd;float t_wd;float* t_xd;float t_yd;float4 rr;float t_zd;float4 rr_d;float t_ae;float t_be;float t_ce;float t_de;float* t_ee;float* t_fe;float* t_ge;float* t_he;
	t_kd=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_kd<__param.__thread_size)){
		t_ld=(float2*)((__param.d + (t_kd<<3)));
		t_md=*t_ld;
		t_nd=(float2*)(((__param.d + (__param.SPAPstride<<3)) + (t_kd<<3)));
		t_od=*t_nd;
		t_pd=(float*)(((__param.d + (__param.SPAPstride<<4)) + (t_kd<<2)));
		t_qd=*t_pd;
		t_rd=(float*)(((__param.d + (__param.SPAPstride*20)) + (t_kd<<2)));
		t_sd=*t_rd;
		t_td=(float*)(((__param.d + (__param.SPAPstride*24)) + (t_kd<<2)));
		t_ud=*t_td;
		t_vd=(float*)(((__param.d + (__param.SPAPstride*36)) + (t_kd<<2)));
		t_wd=*t_vd;
		t_xd=(float*)(((__param.d + (__param.SPAPstride*48)) + (t_kd<<2)));
		t_yd=*t_xd;
		(__param.m_gd)[__param.i]=__param.ret;
		*&rr=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,__param.i);;
		t_zd=((t_sd*t_qd)/ rr.x);
		*&rr_d=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,__param.i);;
		t_ae=((-t_zd*t_wd)/ rr_d.x);
		t_be=(t_zd*((-cosf((t_md).y)*cosf((t_od).y))*sinf(((t_md).x- (t_od).x))));
		t_ce=(t_zd*((cosf((t_md).y)*sinf((t_od).y))- ((sinf((t_md).y)*cosf((t_od).y))*cosf(((t_md).x- (t_od).x)))));
		t_de=(t_ud*t_yd);
		(__param.m_hd)[t_kd]=t_de;
		t_ee=(float*)(((__param.d + (__param.SPAPstride<<4)) + (t_kd<<2)));
		*t_ee=t_de;
		t_fe=(float*)(((__param.d + (__param.SPAPstride*20)) + (t_kd<<2)));
		*t_fe=t_ae;
		t_ge=(float*)(((__param.d + (__param.SPAPstride*24)) + (t_kd<<2)));
		*t_ge=t_be;
		t_he=(float*)(((__param.d + (__param.SPAPstride*28)) + (t_kd<<2)));
		*t_he=t_ce;
	}
}
extern "C" __global__ void CUDAkrnl_697a(struct g_cu_hd __param){int t_od;float2* t_pd;float2 t_qd;float2* t_rd;float2 t_sd;float* t_td;float t_ud;float* t_vd;float t_wd;float* t_xd;float t_yd;float* t_zd;float t__d;float* t_ae;float t_be;float* t_ce;float t_de;float* t_ee;float t_fe;float* t_ge;float t_he;float4 rr;float t_ke;float4 rrle;float t_me;float t_ne;float t_oe;float t_pe;float* t_qe;float* t_re;float* t_se;float* t_te;
	t_od=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_od<__param.__thread_size)){
		t_pd=(float2*)((__param.d + (t_od<<3)));
		t_qd=*t_pd;
		t_rd=(float2*)(((__param.d + (__param.SPAPstride<<3)) + (t_od<<3)));
		t_sd=*t_rd;
		t_td=(float*)(((__param.d + (__param.SPAPstride<<4)) + (t_od<<2)));
		t_ud=*t_td;
		t_vd=(float*)(((__param.d + (__param.SPAPstride*20)) + (t_od<<2)));
		t_wd=*t_vd;
		t_xd=(float*)(((__param.d + (__param.SPAPstride*24)) + (t_od<<2)));
		t_yd=*t_xd;
		t_zd=(float*)(((__param.d + (__param.SPAPstride*28)) + (t_od<<2)));
		t__d=*t_zd;
		t_ae=(float*)(((__param.d + (__param.SPAPstride*36)) + (t_od<<2)));
		t_be=*t_ae;
		t_ce=(float*)(((__param.d + (__param.SPAPstride*40)) + (t_od<<2)));
		t_de=*t_ce;
		t_ee=(float*)(((__param.d + (__param.SPAPstride*44)) + (t_od<<2)));
		t_fe=*t_ee;
		t_ge=(float*)(((__param.d + (__param.SPAPstride*48)) + (t_od<<2)));
		t_he=*t_ge;
		(__param.m_md)[(__param.m_ld+ __param.i)]=__param.ret;
		*&rr=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,__param.i);;
		t_ke=((t_ud*t_de)/ rr.x);
		*&rrle=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,__param.i);;
		t_me=(t_wd+ ((-t_ke*t_be)/ rrle.x));
		t_ne=(t_yd+ (t_ke*((-cosf((t_qd).y)*cosf((t_sd).y))*sinf(((t_qd).x- (t_sd).x)))));
		t_oe=(t__d+ (t_ke*((cosf((t_qd).y)*sinf((t_sd).y))- ((sinf((t_qd).y)*cosf((t_sd).y))*cosf(((t_qd).x- (t_sd).x))))));
		t_pe=(t_fe*t_he);
		(__param.m_nd)[t_od]=t_pe;
		t_qe=(float*)(((__param.d + (__param.SPAPstride<<4)) + (t_od<<2)));
		*t_qe=t_pe;
		t_re=(float*)(((__param.d + (__param.SPAPstride*20)) + (t_od<<2)));
		*t_re=t_me;
		t_se=(float*)(((__param.d + (__param.SPAPstride*24)) + (t_od<<2)));
		*t_se=t_ne;
		t_te=(float*)(((__param.d + (__param.SPAPstride*28)) + (t_od<<2)));
		*t_te=t_oe;
	}
}
extern "C" __global__ void CUDAkrnl_697b(struct g_cu_kd __param){int t_rd;float2* t_sd;float2 t_td;float2* t_ud;float2 t_vd;float* t_wd;float t_xd;float* t_yd;float t_zd;float* t__d;float t_ae;float* t_be;float t_ce;float* t_de;float t_ee;float* t_fe;float t_ge;float4 rr;float t_he;float4 rrke;float t_le;float t_me;
	t_rd=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_rd<__param.__thread_size)){
		t_sd=(float2*)((__param.d + (t_rd<<3)));
		t_td=*t_sd;
		t_ud=(float2*)(((__param.d + (__param.SPAPstride<<3)) + (t_rd<<3)));
		t_vd=*t_ud;
		t_wd=(float*)(((__param.d + (__param.SPAPstride<<4)) + (t_rd<<2)));
		t_xd=*t_wd;
		t_yd=(float*)(((__param.d + (__param.SPAPstride*20)) + (t_rd<<2)));
		t_zd=*t_yd;
		t__d=(float*)(((__param.d + (__param.SPAPstride*24)) + (t_rd<<2)));
		t_ae=*t__d;
		t_be=(float*)(((__param.d + (__param.SPAPstride*28)) + (t_rd<<2)));
		t_ce=*t_be;
		t_de=(float*)(((__param.d + (__param.SPAPstride<<5)) + (t_rd<<2)));
		t_ee=*t_de;
		t_fe=(float*)(((__param.d + (__param.SPAPstride*36)) + (t_rd<<2)));
		t_ge=*t_fe;
		(__param.m_nd)[((__param.m_md<<1)+ __param.i)]=__param.ret;
		*&rr=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,__param.i);;
		t_he=((t_xd*t_ee)/ rr.x);
		*&rrke=tex1Dfetch(*(texture<float4,1,cudaReadModeElementType>*)&__tex0,__param.i);;
		t_le=(t_ae+ (t_he*((-cosf((t_td).y)*cosf((t_vd).y))*sinf(((t_td).x- (t_vd).x)))));
		t_me=(t_ce+ (t_he*((cosf((t_td).y)*sinf((t_vd).y))- ((sinf((t_td).y)*cosf((t_vd).y))*cosf(((t_td).x- (t_vd).x))))));
		(__param.m_od)[t_rd]=(t_zd+ ((-t_he*t_ge)/ rrke.x));
		(__param.m_pd)[t_rd]=t_le;
		(__param.m_qd)[t_rd]=t_me;
	}
}
extern "C" __global__ void CUDAkrnl_703(struct g_cu_ld __param){int t_rd;
	t_rd=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_rd<__param.__thread_size)){
		(__param.d)[(__param.m_md+ __param.i)]=__param.ret;
		(__param.d)[(__param.m_nd+ __param.i)]=__param.m_od;
		(__param.d)[((__param.m_nd+ __param.m_pd)+ __param.i)]=__param.m_qd;
	}
}
extern "C" __global__ void CUDAkrnl_718(struct g_cu_md __param){int t_td;int r;int isi2;int p;int q;int u;int uae;float t_be;float t_ce;float rf;int uhe;
	t_td=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_td<__param.__thread_size)){
		r=0;
		isi2=1;
		p=__param.m_nd;
		q=__param.m_od;
		if((__param.m_pd<62)){
			if((__param.m_pd==37))goto g_cu_ud;if((__param.m_pd==38))goto g_cu_vd;if(!((__param.m_pd==60)))goto g_cu_wd;}else{
			if((__param.m_pd==62))goto g_cu_xd;if(!((__param.m_pd==94))){
				if(!((__param.m_pd==124)))goto g_cu_wd;goto g_cu_yd;g_cu_vd:;r=(p&q);
				goto g_cu_zd;g_cu_yd:;r=(p|q);
				goto g_cu_zd;}
			r=(p^q);
			goto g_cu_zd;}
		r=(p<<q);
		goto g_cu_zd;g_cu_xd:;r=(p>>q);
		goto g_cu_zd;g_cu_ud:;r=(p%q);
		g_cu_wd:;if((__param.m_qd!=0)){
			if((__param.m_rd!=0)){
				if(!((__param.m_pd==47)))goto g_cu__d;if(!((q==0)))goto g_cu__d;}
		}
		if((__param.m_qd!=0)){
			u=__float_as_int((float)p);
			p=u;
		}
		if((__param.m_rd!=0)){
			uae=__float_as_int((float)q);
			q=uae;
		}
		t_be=__int_as_float(p);
		t_ce=__int_as_float(q);
		isi2=0;
		if((__param.m_pd<45)){
			if((__param.m_pd==42))goto g_cu_de;if(!((__param.m_pd==43)))goto g_cu_ee;}else{
			if((__param.m_pd==45))goto g_cu_fe;if(!((__param.m_pd==47)))goto g_cu_ee;goto g_cu_ge;}
		rf=(t_be+ t_ce);
		goto g_cu_ee;g_cu_fe:;rf=(t_be- t_ce);
		goto g_cu_ee;g_cu_de:;rf=(t_be*t_ce);
		goto g_cu_ee;g_cu_ge:;rf=(t_be/ t_ce);
		g_cu_ee:;uhe=__float_as_int(rf);
		r=uhe;
		goto g_cu_zd;g_cu__d:;isi2=1;
		if((__param.m_pd<45)){
			if((__param.m_pd==42))goto g_cu_ke;if(!((__param.m_pd==43)))goto g_cu_zd;}else{
			if((__param.m_pd==45))goto g_cu_le;if(!((__param.m_pd==47)))goto g_cu_zd;goto g_cu_me;}
		r=(p+ q);
		goto g_cu_zd;g_cu_le:;r=(p- q);
		goto g_cu_zd;g_cu_ke:;r=(p*q);
		goto g_cu_zd;g_cu_me:;r=(p/ q);
		g_cu_zd:;*__param.m_sd=r;
		(__param.m_sd)[1]=isi2;
	}
}
extern "C" __global__ void CUDAkrnl_749(struct g_cu_nd __param){int t_sd;int2 t_td;float2 t_ud;float2 t_vd;float4* t_wd;float4 color;float t_xd;float t_yd;float t_zd;float t__d;float t_ae;float t_be;float4 rr;float4* t_ce;float4 t_de;float4 t_ee;int c;
	t_sd=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_sd<__param.__thread_size)){
		t_td=make_int2((t_sd%__param.m_od),(t_sd/ __param.m_od));
		t_ud=make_float2((((float)(t_td).x/ (float)__param.m_od)*3.60000000e+002f),((((float)(t_td).y/ (float)__param.m_pd)*1.80000000e+002f)+ -9.00000000e+001f));
		t_vd=make_float2(((t_ud).x*1.74532905e-002f),((t_ud).y*1.74532905e-002f));
		t_wd=&color;
		t_xd=((t_vd).x/ 6.28318453e+000f);
		if((t_xd<1.53000012e-001f)){
			t_yd=__int_as_float(0);
		}else{
			if((t_xd>3.97000015e-001f)){
				t_zd=1.00000000e+000f;
			}else{
				t_zd=((t_xd- 1.53000012e-001f)/ 2.44000003e-001f);
			}
			t_yd=t_zd;
		}
		t__d=(((t_vd).y+ 1.57079613e+000f)/ 3.14159226e+000f);
		if((t__d<3.23000014e-001f)){
			t_ae=__int_as_float(0);
		}else{
			if((t__d>7.73000002e-001f)){
				t_be=1.00000000e+000f;
			}else{
				t_be=((t__d- 3.23000014e-001f)/ 4.49999988e-001f);
			}
			t_ae=t_be;
		}
		*&rr=tex2D(*(texture<float4,2,cudaReadModeElementType>*)&__tex0,t_yd,t_ae);;
		t_ce=&rr;
		t_de=*t_ce;
		t_ee=make_float4((t_de).x,(t_de).y,(t_de).z,(t_de).w);
		*t_wd=t_ee;
		c=0;
		goto g_cu_he;g_cu_ne:;(__param.m_rd)[((__param.m_qd*t_sd)+ c)]=*(float*)(((char*)(&color)+ (c<<2)));
		c=(c+ 1);
		g_cu_he:;if((c<__param.m_qd))goto g_cu_ne;}
}
extern "C" __global__ void CUDAkrnl_826(struct g_cu_od __param){int t_rd;int r;float t_sd;float t_de;float t_ee;
	t_rd=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_rd<__param.__thread_size)){
		r=__param.a;
		t_sd=*(float*)((&__param.a));
		if((__param.c<108)){
			if((__param.c<102)){
				if((__param.c==99))goto g_cu_td;if(!((__param.c==101)))goto g_cu_ae;goto g_cu_be;}
			if((__param.c==102))goto g_cu_ce;if(!((__param.c==105)))goto g_cu_ae;}else{
			if(!((__param.c==108))){
				if(!((__param.c==115))){
					if(!((__param.c==116)))goto g_cu_ae;}else{
					r=__float_as_int(sinf(t_sd));
					goto g_cu_ae;g_cu_td:;r=__float_as_int(cosf(t_sd));
					goto g_cu_ae;}
				r=__float_as_int(tanf(t_sd));
				goto g_cu_ae;}
			t_de=log(t_sd);
			r=__float_as_int(t_de);
			goto g_cu_ae;g_cu_be:;t_ee=exp(t_sd);
			r=__float_as_int(t_ee);
			goto g_cu_ae;}
		if(!((__param.m_pd!=0))){
			r=(int)t_sd;
			goto g_cu_ae;g_cu_ce:;if((__param.m_pd!=0)){
				r=__float_as_int((float)__param.a);
				g_cu_ae:;}
		}
		*__param.m_qd=r;
	}
}
extern "C" __global__ void CUDAkrnl_835(struct g_cu_pd __param){int t_td;int2 t_ud;float2 t_vd;float2 t_wd;float4* t_xd;float4 color;float t_yd;float t_zd;float t__d;float t_ae;float t_be;float t_ce;float4 rr;float4* t_de;float4 t_ee;float4 t_fe;int c;
	t_td=blockIdx.x*blockDim.x+threadIdx.x;
	if((t_td<__param.__thread_size)){
		t_ud=make_int2((t_td%__param.m_qd),(t_td/ __param.m_qd));
		t_vd=make_float2((((float)(t_ud).x/ (float)__param.m_qd)*3.60000000e+002f),((((float)(t_ud).y/ (float)__param.m_rd)*1.80000000e+002f)+ -9.00000000e+001f));
		t_wd=make_float2(((t_vd).x*1.74532905e-002f),((t_vd).y*1.74532905e-002f));
		t_xd=&color;
		t_yd=((t_wd).x/ 6.28318453e+000f);
		if((t_yd<1.53000012e-001f)){
			t_zd=__int_as_float(0);
		}else{
			if((t_yd>3.97000015e-001f)){
				t__d=1.00000000e+000f;
			}else{
				t__d=((t_yd- 1.53000012e-001f)/ 2.44000003e-001f);
			}
			t_zd=t__d;
		}
		t_ae=(((t_wd).y+ 1.57079613e+000f)/ 3.14159226e+000f);
		if((t_ae<3.23000014e-001f)){
			t_be=__int_as_float(0);
		}else{
			if((t_ae>7.73000002e-001f)){
				t_ce=1.00000000e+000f;
			}else{
				t_ce=((t_ae- 3.23000014e-001f)/ 4.49999988e-001f);
			}
			t_be=t_ce;
		}
		*&rr=tex2D(*(texture<float4,2,cudaReadModeElementType>*)&__tex0,t_zd,t_be);;
		t_de=&rr;
		t_ee=*t_de;
		t_fe=make_float4((t_ee).x,(t_ee).y,(t_ee).z,(t_ee).w);
		*t_xd=t_fe;
		c=0;
		goto g_cu_oe;g_cu_pe:;(__param.m_sd)[((__param.nc*t_td)+ c)]=*(float*)(((char*)(&color)+ (c<<2)));
		c=(c+ 1);
		g_cu_oe:;if((c<__param.nc))goto g_cu_pe;}
}
