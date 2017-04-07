// SRBFGen.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "tool.h"
#include "srbf.h"

#define C_PI			3.1415926535897932384626433832795f
#define C_D2R			(C_PI/180.f)

#pragma warning ( disable : 4996 )
#pragma warning ( disable : 4267 )
#pragma warning ( disable : 4018 )
#pragma warning ( disable : 4819 )

//#include <scitbx/lbfgsb/raw.h>
//#include <scitbx/array_family/shared.h>
//#include <scitbx/lbfgsb.h>

#pragma warning ( default : 4996 )
#pragma warning ( default : 4267 )
#pragma warning ( default : 4018 )
#pragma warning ( default : 4819 )

#include "sys/rconsole.h"

#include <conio.h>
#include <windows.h>

using scitbx::af::shared;
using namespace scitbx::lbfgsb::raw;

enum COMPONENT{
	COMPONENT_WR,
	COMPONENT_WG,
	COMPONENT_WB,
	COMPONENT_CX,
	COMPONENT_CY,
	COMPONENT_LD
};

template <typename ElementType>
ref1<ElementType>
make_ref1(shared<ElementType>& a)
{
	return ref1<ElementType>(a.begin(), int(a.size()));
}

struct Config
{
	char _szProbeFilename[260];
	char _szInitialSRBFFilename[260];
	char _szOptSRBFFilename[260];
	char _szProbeAnimationFNTemplate[260];
	float _minLambda, _maxLambda;
	int _startFrame, _endFrame;
	int _numSRBFs;
	int _teleportInterval;
	bool _bExportDirLights;

	Config()
	{
		_szProbeFilename[0]='\0';
		_szInitialSRBFFilename[0]='\0';
		_szOptSRBFFilename[0]='\0';
		_szProbeAnimationFNTemplate[0]='\0';
		_numSRBFs=0;
		_teleportInterval=20;
		_minLambda=_maxLambda=1.f;
		_startFrame=_endFrame=0;
		_bExportDirLights=false;
	}

	bool Read(const char* fn)
	{
		FILE* f=NULL;
		errno_t err=fopen_s(&f,fn,"rt");
		if(err!=0)
		{
			ERRBEGIN;
			printf("Error opening %s.\n",fn);					
			ERREND;		
			return false;
		}
		char token[256];		
		while(fscanf_s(f,"%s",token,256)!=EOF)
		{
			if('#'==token[0])	//comments
			{
				char c=fgetc(f);
				while(c!='\n'&&c!=EOF) c=fgetc(f);
				continue;
			}else if(strcmp(token,"NumSRBFs")==0)
			{
				if(fscanf_s(f,"%d",&_numSRBFs)!=1||_numSRBFs<=0)				
				{
					console::Message("Error reading SRBF number.\n",console::MSGTYPE_ERR);					
					fclose(f);
					return false;
				}
			}else if(strcmp(token,"TeleportInterval")==0)
			{
				if(fscanf_s(f,"%d",&_teleportInterval)!=1)				
				{
					console::Message("Error reading teleport interval.\n",console::MSGTYPE_ERR);					
					fclose(f);
					return false;
				}
			}else if(strcmp(token,"ProbeFilename")==0)
			{
				if(fscanf_s(f,"%s",_szProbeFilename,255)!=1)
				{
					console::Message("Error reading probe filename.\n",console::MSGTYPE_ERR);										
					fclose(f);
					return false;
				}			
			}else if(strcmp(token,"OptSRBFFilename")==0)
			{
				if(fscanf_s(f,"%s",_szOptSRBFFilename,255)!=1)
				{
					console::Message("Error reading output filename.\n",console::MSGTYPE_ERR);										
					fclose(f);
					return false;
				}			
			}else if(strcmp(token,"ProbeAnimation")==0)
			{
				if(fscanf_s(f,"%s",_szProbeAnimationFNTemplate,255)!=1)
				{
					console::Message("Error reading probe animation filename template.\n",console::MSGTYPE_ERR);										
					fclose(f);
					return false;
				}			
				if(fscanf_s(f,"%d %d",&_startFrame,&_endFrame)!=2)
				{
					console::Message("Error reading start/end frame#.\n",console::MSGTYPE_ERR);
					fclose(f);
					return false;
				}
			}else if(strcmp(token,"InitialSRBFFilename")==0)
			{
				if(fscanf_s(f,"%s",_szInitialSRBFFilename,255)!=1)
				{
					console::Message("Error reading initlial srbf filename.\n",console::MSGTYPE_ERR);										
					fclose(f);
					return false;
				}			
			}else if(strcmp(token,"LambdaMinMax")==0)
			{
				if(fscanf_s(f,"%f %f",&_minLambda, &_maxLambda)!=2)				
				{
					console::Message("Error reading lambda minmax.\n",console::MSGTYPE_ERR);					
					fclose(f);
					return false;
				}
			}else if(strcmp(token,"ExportDirLights")==0)
			{
				_bExportDirLights=true;
			}else
			{				
				ERRBEGIN;
				printf("Unknown indicator - %s\n",token);
				ERREND;
				fclose(f);
				return false;
			}
		}
		fclose(f);
		return true;
	}
} cfg;

int __cdecl fitting_main_loop();
int __cdecl fitting_main_loop_seq();

int main_initializer(int argc, char* argv[])
{
	if( argc < 3 )
	{
		printf( "usage: intializer <number of srbfs> <initialization_filename>\n" );
		return -1;
	}		
	int n = 0;
	if( sscanf_s( argv[1], "%d", &n ) != 1 )
	{
		printf( "bad input srbf number(%s).\n", argv[1] );
		return -2;
	}

	random_initialize_clear( n );
	output_srbfs( argv[2] );
	printf( "initial srbf saved to %s.\n", argv[2] );
	return 0;
}

_inline double get_component( ref1<double>& ref, int i, COMPONENT c )
{
	int n = get_srbf_number();
	return ref( c * n + i );
}


void write_opt_param(ref1<double>& x_ref)
{
	srbf* srbfs = (srbf*)get_srbfs();
	int n = get_srbf_number();
	for( int i=1; i<=n; i++ )
	{	
		int j = i - 1;
		
		x_ref(i) = srbfs[j]._weight[0];
		x_ref(n + i) = srbfs[j]._weight[1];
		x_ref(2 * n + i) = srbfs[j]._weight[2];
		
		x_ref(3 * n + i) = srbfs[j]._center[0];
		x_ref(4 * n + i) = srbfs[j]._center[1];
		
		x_ref(5 * n + i) = srbfs[j]._lambda;
	}
}

void read_opt_param(const ref1<double>& x_ref)
{
	srbf* srbfs = (srbf*)get_srbfs();
	int n = get_srbf_number();
	for( int i=1; i<=n; i++ )
	{	
		int j = i - 1;
		
		srbfs[j]._weight[0] = (float)x_ref(i);
		srbfs[j]._weight[1] = (float)x_ref(n + i);
		srbfs[j]._weight[2] = (float)x_ref(2 * n + i);
		
		srbfs[j]._center[0] = (float)x_ref(3 * n + i);
		srbfs[j]._center[1] = (float)x_ref(4 * n + i);
		
		srbfs[j]._lambda = (float)x_ref(5 * n + i);
	}
}

int __cdecl fitting_main_loop()
{
	using scitbx::af::shared;
	using scitbx::fn::pow2;
	using namespace scitbx::lbfgsb::raw;
	using namespace scitbx::lbfgsb;

	int n = get_srbf_number();
	shared<double>	x(6 * n);
	ref1<double>	x_ref = make_ref1(x);		
	shared<double>	l(6 * n);
	shared<double>	u(6 * n);	
	shared<double>	g(6 * n);
	ref1<double>	g_ref = make_ref1(g);		
	shared<int>		nbd(6 * n);

	//bounds for weights
	for( int i=0; i<3*n; i++ )
	{
		nbd[i] = 2;
		l[i] = 0.f;
		u[i] = 10000.f;
	}

	for( int i=0; i<n; i++ )
	{
		//bounds for centers
		nbd[3 * n + i] = nbd[4 * n + i] = 2;		
		l[3 * n + i] = 0;
		l[4 * n + i] = -C_PI / 2;
		u[3 * n + i] = 2 * C_PI;
		u[4 * n + i] = C_PI / 2;		
		
		//bounds for lambda
		nbd[5 * n + i] = 2;
		l[5 * n + i] = 1.f/cfg._maxLambda;			
		u[5 * n + i] = 1.f/cfg._minLambda;		
	}

	for( int i=1; i<=n; i++ )
	{
		g_ref(i) = 0;
	}

		
	minimizer<double> srbf_minimizer( 6 * n, 17, l, u, nbd, 1.0e+9, 1.0e-5, -1 );			

	try{			
		double f = evaluate( NULL );
		printf( "f=%lf\n", f );		
		printf(
			 "\n"
			 "     Solving optimal parameters...\n"
			 "\n");		

		int nit = 0;						
		
		bool proceed = true;				
		while(proceed)
		{
			write_opt_param( x_ref );
			srbf_minimizer.request_restart();							
			int teleport_counter = 0;
			const int teleport_interval = cfg._teleportInterval;
			while(true)
			{					
				if( srbf_minimizer.process( x_ref, f, g_ref ) )
				{	
					printf( "f=%lf\n", f );		
					read_opt_param( x_ref );
					f = evaluate( &g_ref(1) );						
					if( teleport_interval > 0 && ++teleport_counter==teleport_interval )
					{
						teleport_counter = 0;
						if(teleport())
						{							
							write_opt_param( x_ref );
							srbf_minimizer.request_restart();
							break;
						}
					}					
				}else if(srbf_minimizer.is_terminated())
				{			
					printf( "done. MSE=%5.3f%%.\n", rse()*100.f  );
					proceed = false;
					break;					
				}
				if( cfg._bExportDirLights&&!output_dirs(cfg._szOptSRBFFilename)||!cfg._bExportDirLights&&!output_srbfs( cfg._szOptSRBFFilename ) ) printf( "error output srbfs." );
			}			
		}		
		read_opt_param( x_ref );
		
						
	}catch(std::exception const& e) {
		printf( "%s\n", e.what() );
	}

	return 1;
}

int __cdecl fitting_main_loop_seq()
{
	using scitbx::af::shared;
	using scitbx::fn::pow2;
	using namespace scitbx::lbfgsb::raw;
	using namespace scitbx::lbfgsb;

	int n = get_srbf_number();
	shared<double>	x(6 * n);
	ref1<double>	x_ref = make_ref1(x);		
	shared<double>	l(6 * n);
	shared<double>	u(6 * n);	
	shared<double>	g(6 * n);
	ref1<double>	g_ref = make_ref1(g);		
	shared<int>		nbd(6 * n);

	//bounds for weights
	for( int i=0; i<3*n; i++ )
	{
		nbd[i] = 2;
		l[i] = 0.f;
		u[i] = 10000.f;
	}

	for( int i=0; i<n; i++ )
	{
		//bounds for centers
		nbd[3 * n + i] = nbd[4 * n + i] = 2;
		l[3 * n + i] = 0.238f*2*C_PI;
		l[4 * n + i] = 0.373f*C_PI-C_PI/2;
		u[3 * n + i] = 0.312f*2*C_PI;
		u[4 * n + i] = 0.733f*C_PI-C_PI/2;		
		
		//bounds for lambda
		nbd[5 * n + i] = 2;
		l[5 * n + i] = 1.f/cfg._maxLambda;			
		u[5 * n + i] = 1.f/cfg._minLambda;		
	}

	for( int i=1; i<=n; i++ )
	{
		g_ref(i) = 0;
	}

	FILE* output = NULL;
	int frames = cfg._endFrame - cfg._startFrame + 1;
	if( frames<=0 )
	{
		console::Message("bad start/end frame#.\n",console::MSGTYPE_ERR);
		return -1;
	}
	errno_t err = fopen_s( &output, cfg._szOptSRBFFilename, "wt" );
	if(err!=0)
	{
		console::Message("failed open output file.\n",console::MSGTYPE_ERR);
		return -2;
	}	
	fprintf_s( output, "%d %d\n", frames, cfg._numSRBFs );
		
	minimizer<double> srbf_minimizer( 6 * n, 17, l, u, nbd, 1.0e+9, 1.0e-5, -1 );		
	try{			
		int i=cfg._startFrame;
		while(1)
		{			
			char pfm_fn[MAX_PATH];
			//sprintf_s(srbf_fn,"%d.srbf",i);			
			double f = evaluate( NULL );
			printf( "f=%lf\n", f );		
			printf(
				 "\n"
				 "     Solving optimal parameters...\n"
				 "\n");		

			int nit = 0;						
			
			bool proceed = true;				
			while(proceed)
			{
				write_opt_param( x_ref );
				srbf_minimizer.request_restart();							
				int teleport_counter = 0;
				const int teleport_interval = (i==1)?cfg._teleportInterval:0;
				while(true)
				{					
					if( srbf_minimizer.process( x_ref, f, g_ref ) )
					{	
						printf( "f=%lf\n", f );		
						read_opt_param( x_ref );
						f = evaluate( &g_ref(1) );						
						if( teleport_interval > 0 && ++teleport_counter==teleport_interval )
						{
							teleport_counter = 0;
							if(teleport())
							{							
								write_opt_param( x_ref );
								srbf_minimizer.request_restart();
								break;
							}
						}					
					}else if(srbf_minimizer.is_terminated())
					{			
						printf( "done. MSE=%5.3f%%.\n", rse()*100.f  );						
						proceed = false;
						if( !append_srbfs( output ) ) printf( "error output srbfs." );
						break;					
					}
					//if( !output_srbfs( srbf_fn ) ) printf( "error output srbfs." );
				}			
			}		
			read_opt_param( x_ref );
			i++;
			if(i>cfg._endFrame)break;
			sprintf_s(pfm_fn,cfg._szProbeAnimationFNTemplate,i);
			update_envlight(pfm_fn);
		}		
						
	}catch(std::exception const& e) {
		printf( "%s\n", e.what() );
	}

	fclose(output);

	return 1;
}

int main(int argc, char* argv[])
{	
	if( argc != 2 )
	{
		printf( "usage: SRBFGen <config_filename>\n" );
		return -1;
	}			
	if(!cfg.Read(argv[1])) return -1;
	bool animation_fit = false;
	char fn_probe[260];
	if(strlen(cfg._szProbeAnimationFNTemplate)!=0)
	{
		set_textured();
		animation_fit = true;
		sprintf_s(fn_probe,260,cfg._szProbeAnimationFNTemplate,cfg._startFrame);
	}else
	{
		strcpy_s(fn_probe,260,cfg._szProbeFilename);
	}
	if(cfg._numSRBFs>0)
	{
		if(strlen(cfg._szInitialSRBFFilename)==0)
		{
			printf("output initialization srbf filename expected.\n");
			return -2;
		}
		//if(animation_fit)
		//	random_initialize( cfg._numSRBFs );
		//else
		//	random_initialize_clear( cfg._numSRBFs );
		random_initialize_clear( cfg._numSRBFs );
		output_srbfs( cfg._szInitialSRBFFilename );
		MSGBEGIN;
		printf( "initial srbf saved to %s.\n", cfg._szInitialSRBFFilename );
		MSGEND;
		return true;
	}else if(strlen(cfg._szInitialSRBFFilename)==0)
	{
		printf("input initialization srbf filename expected.\n");
		return -2;
	}else
	{
		if( init_pfm( fn_probe, cfg._szInitialSRBFFilename, animation_fit?fitting_main_loop_seq:fitting_main_loop)<0 )
		{
			printf( "failed init pfm file %s\n", argv[1] );
			return -3;
		}
	}	
	return 0;
}