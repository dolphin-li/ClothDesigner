uses gl, cubsgp, bsgpdbg, coff, woo
///cuda.enable_checkpoint=1

#include "srbf.h"

#define C_PI			3.1415926535897932384626433832795f
#define C_D2R			(C_PI/180.f)
#define MAX_NUM_SRBF	500

textured=0;
const texleft=0.153f;
const texright=0.397f;
const texbottom=0.323f;
const textop=0.773f;
int texwidth,texheight;

texture(float4, 2, cudaReadModeElementType) tex;

static timer
	u64 freq
	__init__()
		QueryPerformanceFrequency(LARGE_INTEGER*&.freq)
	%time()
		QueryPerformanceCounter(LARGE_INTEGER*&u64 a);
		return double a/double(.freq)

typedef int()() FittingProc;

/*----------------------------------------------------
for all 2d coordinates used in this file:
.x corresponds to the longitude, ranges from 0 to C_PI * 2 in ll space, and from 0 to ._w -1 in image space
.y corresponds to the lattitude, ranges from -C_PI/2 to C_PI/2
for the derivative array, in which:
0-3n-1 are for weights
3n-5n-1 are for centers
5n- are for lambda
NOTE: we choose to opt for 1.f/lambda, because it linearizes better.
------------------------------------------------------*/

//lattitute/longtitude coordinate to world coordinate
float2 ll2world( float2 lcoord ) = make_float2( lerp( -1.f, 1.f, lcoord.x * 0.5f / C_PI ), lerp( -1.f, 1.f, (lcoord.y + C_PI / 2.f) / C_PI ) );

__both__ float gdist( float2 l1, float2 l2 ) = cos(l1.y) * cos(l2.y) * cos(l1.x - l2.x) + sin(l1.y) * sin(l2.y);
__both__ float dGDdx( float2 l1, float2 l2 ) 
	return -cos(l1.y) * cos(l2.y) * sin(l1.x - l2.x);
__both__ float dGDdy( float2 l1, float2 l2 ) 
	return cos(l1.y) * sin(l2.y) - sin(l1.y) * cos(l2.y) * cos(l1.x - l2.x);

__both__ float energy( srbf& b ) = (1.f-exp(-2.f/b._lambda))*b._lambda*2.f*C_PI;
__both__ float srbf_eval( float2 center, float lambda, float2 omega ) = exp((gdist( omega, center ) - 1.f) / lambda);
float rse();
void fill();
void _output_pfm( char* path, int w, int h, int nc, float* raster )	;

__both__ float smoothstep(a,b,x)
	return (x<a)?0.f:((x>b)?1.f:lerp(0.f,1.f,(x-a)/(b-a)));

__both__ int inside_box(float2 coord)
	coord0=coord;
	coord0.x/=2.f*PI;
	coord0.y=(coord0.y+PI/2.f)/PI;
	return coord0.x<texright&&coord0.x>texleft&&coord0.y<textop&&coord0.y>texbottom;

__device__ float4 get_tex_raster(float2 coord)
	texcoordx=smoothstep(texleft,texright,(coord.x/(2.f*PI)));
	texcoordy=smoothstep(texbottom,textop,(coord.y+PI/2.f)/PI);
	return tex2D($tex,texcoordx,texcoordy);	

class pfm
	int		_w, _h;																				//width/height of the lattitude/longtitude map
	int		_nc;
	float*	_raster;	
	dlist(float) _rasterD;
	dlist(float) _approxD;	
	CUarray _ta;	

	__init__()
		._w = ._h = 0;
		._nc = 0;
		._raster = NULL;	
		._ta=NULL;
		._rasterD = new float<>;
		._approxD = new float<>;
		return;

	__done__()
		if( ._raster ):	delete ._raster;
		if( ._rasterD ): ._rasterD.discard();
		if( ._approxD ): ._approxD.discard();
		return;		

	//image coordinate to lattitute/longtitude coordinate
	__both__ float2 image2ll( int2 icoord ) = make_float2( lerp( 0.f, 360.f, float(icoord.x)/float(._w) ), lerp( -90.f, 90.f, float(icoord.y)/float(._h) ) ) * C_D2R;			

	__both__ float2 pid2ll( int id ) = .image2ll( make_int2(id % ._w, id / ._w) );

	__both__ int2 ll2image( float2 lcoord ) = make_int2( (int)(lcoord.x * float(._w) * 0.5f / C_PI ), (int)((lcoord.y + C_PI / 2.f) * float(._h) / C_PI) );

	void accumulate( srbf& b )
		assert( ._approxD && ._w > 0 && ._h > 0 && ._nc == 3 );
		approxL = ._approxD;
		spawn ._w * ._h
			tid = thread.rank;
			x = tid % ._w;
			y = tid / ._w;
			float3 val = srbf_eval( b._center, b._lambda, .image2ll(make_int2(x, y)) ) * b._weight;				
			approxL[tid * ._nc] += val.x;
			approxL[tid * ._nc + 1] += val.y;
			approxL[tid * ._nc + 2] += val.z;

	float* pixel( int x, int y ) = &._raster[(y * ._w + x) * ._nc];
	
pfm*			envLight;
dlist(srbf)		initialD;
srbf*			initial = NULL;
srbf*			intermediateD = NULL;
int				initial_number = 0;
FittingProc		loop = NULL;
int				optimizing = 0;
int				init_srbf_num = 20;

objexport set_textured()
	$textured = 1;

int load_pfm( char* name, pfm& out, int tex )		
	int sc = 1;
	char[255] identifier;

	writeln( "load_pfm" );
	
	f =	fopen( name, "rb" );
	if( !f )
		sc = 0;
		return sc;
	fscanf( f, "%s\n", identifier, 255 );		
	writeln( identifier );
	if( strcmp( identifier, "Pf" ) == 0 ): out._nc = 1;
	else if( strcmp( identifier, "PF" ) == 0 ): out._nc = 3;
	else if( strcmp( identifier, "P4" ) == 0 ): out._nc = 4;
	else sc = 0;			
	if( fscanf( f, "%d %d\n", &out._w, &out._h ) != 2 )		
		sc = 0;			
	writeln( 2, out._w, " ", out._h );
	readln( f );	
	if( out._w > 0 && out._h > 0 )		
		int nitems = out._w * out._h;
		out._raster = new float[ nitems * out._nc ];
		if( fread( out._raster, out._nc * sizeof(float), out._w * out._h, f ) != nitems )			
			delete out._raster;			
			sc = 0;
			out._raster = NULL;									
	else
		sc = 0;		
	writeln( 3 );
	
	if( sc )
		out._rasterD.clear_resize( out._w * out._h * out._nc );
		out._approxD.clear_resize( out._w * out._h * out._nc );
		cpyh2d( out._rasterD.d, out._raster, out._w * out._h * out._nc * sizeof(float) );
		nc=out._nc;		
		if tex:		
			if out._ta==NULL: 
				out._ta = danew float4(out._w, out._h);				
				with $tex
					.addressMode[0]=cudaAddressModeClamp;
					.addressMode[1]=cudaAddressModeClamp;
					.filterMode=cudaFilterModeLinear;
					.normalized=1;	
			fth = new[out._w * out._h] float4;
			cuMemsetD32(fth,0,out._w*out._h*4);
			out._ta = danew float4(out._w, out._h);				
			for i=0:out._w*out._h-1				
				p=&out._raster[i*nc];				
				for c=0:nc-1
					fth[i][c]=p[c];						
			cpyh2a(out._ta,fth);
			delete fth
			$tex.array=out._ta;
			$texwidth=out._w;
			$texheight=out._h;
			out._w=512;
			out._h=256;

	writeln( 4 );
	fclose( f );
	return sc;

float integral( srbf& srbf )
	return 2.f*PI*(1.f-exp(-2.f/srbf._lambda))*srbf._lambda*(srbf._weight.x*srbf._weight.x+srbf._weight.y*srbf._weight.y+srbf._weight.z*srbf._weight.z);

int save_dir( char* name, srbf* srbfs, int num )
	buffer = new[num] int;
	for i=0:num-1
		buffer[i]=i;
	/*
	for i=0:num-1
		buffer[7*i]=2.f*PI*(1.f-exp(-2.f/srbfs[i]._lambda))*srbfs[i]._lambda;
		buffer[7*i+3]=sin(srbfs[i]._center.y);
		buffer[7*i+1]=cos(srbfs[i]._center.y)*cos(srbfs[i]._center.x);
		buffer[7*i+2]=-cos(srbfs[i]._center.y)*sin(srbfs[i]._center.x);*/		
			
	for (int i=0;i<num;i++)
		for (j=num-2;j>=i;j--)
			if (integral(srbfs[buffer[j]])<integral(srbfs[buffer[j+1]])):
				tmp=buffer[j];
				buffer[j]=buffer[j+1];
				buffer[j+1]=tmp;

	f=fopen(name,"wt");
	if(!f)
		return 0;
	fprintf(f,"%d\n",num);
	for( int i=0; i<num; i++ )
		y=sin(srbfs[buffer[i]]._center.y);
		x=cos(srbfs[buffer[i]]._center.y)*cos(srbfs[buffer[i]]._center.x);
		z=-cos(srbfs[buffer[i]]._center.y)*sin(srbfs[buffer[i]]._center.x);
		l = sqrt(srbfs[buffer[i]]._weight.x*srbfs[buffer[i]]._weight.x+srbfs[buffer[i]]._weight.y*srbfs[buffer[i]]._weight.y+srbfs[buffer[i]]._weight.z*srbfs[buffer[i]]._weight.z);
		fprintf( f, "%f\t%f\t%f\t%f\t%f\t%f\t%f\n",	srbfs[buffer[i]]._weight.x/l, srbfs[buffer[i]]._weight.y/l, srbfs[buffer[i]]._weight.z/l,
												x,y,z,
												integral(srbfs[buffer[i]]));
	fclose(f);		
	delete buffer;
	return 1;


int save_srbf( char* name, srbf* srbfs, int num )
	int sc = 1;
	f = fopen( name, "wt" );
	if( !f )
		sc = 0;
	fprintf( f, "%d\n", num );
	for( int i=0; i<num; i++ )
		fprintf( f, "%f\t%f\t%f\t%f\t%f\t%f\n",	srbfs[i]._weight.x, srbfs[i]._weight.y, srbfs[i]._weight.z,
												srbfs[i]._center.x, srbfs[i]._center.y,
												1.f / srbfs[i]._lambda );
	fclose( f );
	return sc;

int read_srbf( char* name, srbf* srbfs, int& num )
	int sc = 1;
	f = fopen( name, "rt" );
	if( !f )		
		return 0;
	if (fscanf( f, "%d\n", &num )!=1):sc=0;		
	for( int i=0; i<num; i++ )
		if( fscanf( f, "%f %f %f %f %f %f\n", &(srbfs[i]._weight.x), &(srbfs[i]._weight.y), &(srbfs[i]._weight.z),
												&(srbfs[i]._center.x), &(srbfs[i]._center.y),
												&(srbfs[i]._lambda) ) != 6 ): sc = 0;
		//srbfs[i]._weight = make_float3(0.f, 0.f, 0.f);
		srbfs[i]._lambda = 1.f/srbfs[i]._lambda;

	fclose( f );
	return sc;

int random_initialize( int num_srbf )
	srand( clock() );	
	for ( ; $initial_number<num_srbf; $initial_number++ )
		accept = 0;
		attempts = 0;
		float2 lcoord;
		float* color=NULL;
		while( !accept )
			if( attempts++ > 10000)
				break;
			lcoord = make_float2(lerp( 0.f, 2.f * C_PI, float(rand()) / float(RAND_MAX) ), lerp( -C_PI/2.f, C_PI/2.f, float(rand()) / float(RAND_MAX) ) );		
			int j = 0;
			int2 icoord = $envLight.ll2image( lcoord );
			float b=0.f;
			r = float(rand()) / float(RAND_MAX);						
			if($textured)				
				texcoordx=smoothstep(texleft,texright,(lcoord.x/(2.f*PI)));
				texcoordy=smoothstep(texbottom,textop,(lcoord.y+PI/2.f)/PI);
				nx=(int)floor(texcoordx*float($texwidth-1));
				ny=(int)floor(texcoordy*float($texheight-1));				
				color=&($envLight._raster[(ny*$texwidth+nx)*3]);				
			else
				color = $envLight.pixel(icoord.x, icoord.y);					
			b = sqrt(color[0] * color[0] + color[1] * color[1] + color[2] * color[2]);
			writeln(r," ",b);
			if(r>b)
				continue;			
			for ( ; j<$initial_number; j++ )				
				if( gdist( lcoord, $initial[j]._center ) > cos( C_PI / 16.f ) && r < b )
					continue;
			accept = (j==$initial_number);
		if( !accept )
			writeln( "poisson sampling failed." );
			break;
		$initial[$initial_number]._center = lcoord;			
		$initial[$initial_number]._weight = make_float3( color[0], color[1], color[2] );
		$initial[$initial_number]._lambda = 1.f/64.f;			
	writeln( $initial_number );
	return 1;

objexport int random_initialize_clear( int num_srbf )
	srand( clock() );	
	if $initial == NULL: $initial = new[MAX_NUM_SRBF] srbf;	
	poisson_radius = $textured? C_PI / 48.f : C_PI / 16.f;
	for ( $initial_number=0; $initial_number<num_srbf; $initial_number++ )		
		accept = 0;
		attempts = 0;
		float2 lcoord;		
		while( !accept )
			if( attempts++ > 10000)
				break;
			lcoord = make_float2(lerp( 0.f, 2.f * C_PI, float(rand()) / float(RAND_MAX) ), lerp( -C_PI/2.f, C_PI/2.f, float(rand()) / float(RAND_MAX) ) );		
			if $textured&&!inside_box(lcoord)
				continue;
			int j = 0;
			for ( ; j<$initial_number; j++ )
				if( gdist( lcoord, $initial[j]._center ) > cos( poisson_radius ) )
					break;
			accept = (j==$initial_number);							
		if( !accept )
			writeln( "poisson sampling failed." );
			break;		
		$initial[$initial_number]._center = lcoord;		
		$initial[$initial_number]._weight = make_float3( 0.f, 0.f, 0.f );
		$initial[$initial_number]._lambda = 1.f/48.f;						
	return 1;

class fbWnd:glwindow
	GLuint				tex;
	CGLbuffer			pbo;
	int					nb;																		//number of bytes per channel;
	int					nc;																		//number of channels;
	int					format;
	int					internal_format;
	int					type;
	int					w, h;
	int					show_srbf;

	float* begin_write();
	end_write();
	render();
	update_texture( int l, int t, int w, int h );

	int setup( char* s, int w, int h, int nc, int nb, int isFloat )		
		.title = s;
		.w = w;
		.h = h;
		if w>=GetSystemMetrics(SM_CXSCREEN)||h>=GetSystemMetrics(SM_CYSCREEN):
			.style=WS_POPUP
		with .client_rect
			.w=w;
			.h=h;
		with .rect
			.x = 0;
			.y = 0;
		.nb = nb;
		.nc = nc;
		switch( nb )
			case 4:
				.type = GL_FLOAT;
				break;
			case 1:
				.type = GL_UNSIGNED_BYTE;
				break;
			default:
				assert(0)
				return -1;
		switch( nc )
			case 4:
				.format = GL_RGBA;
				.internal_format = isFloat ? GL_RGBA32F_ARB : GL_RGBA;
				break;
			case 3:
				.format = GL_RGB;
				.internal_format = isFloat ? GL_RGB32F_ARB : GL_RGB;
				break;
			case 1:
				.format = GL_LUMINANCE;
				.internal_format = isFloat ? GL_LUMINANCE32F_ARB : GL_LUMINANCE;
				break;
			default:
				assert(0)
				return -1;
		return 0;

	operator()()		
		Sleep(1);	

	glOnInit()		
		wglSwapIntervalEXT(0);

		bk=.makecurrent()

		.pbo=new CGLbuffer
		.pbo.clear_resize( .w * .h * .nb * .nc );

		glGenTextures( 1, &.tex )
		glBindTexture( GL_TEXTURE_2D, .tex )
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
		glTexImage2D( GL_TEXTURE_2D, 0, .internal_format, .w, .h, 0, .format, .type, null );
		glBindTexture( GL_TEXTURE_2D,0u );

		glMatrixMode( GL_MODELVIEW );
		glLoadIdentity();
		glMatrixMode( GL_PROJECTION );
		glLoadIdentity();

		.unmakecurrent(bk)

		SetForegroundWindow( .hwnd );

	OnLButtonDown(int k,int x,int y)
		if $optimizing:return;		
		y = .h - y;				
		writeln( $initial, " ", $initial_number );
		$initial[$initial_number]._center = make_float2( lerp( 0.f, 360.f, float(x)/float($envLight._w) ), lerp( -90.f, 90.f, float(y)/float($envLight._h) ) ) * C_D2R;//$envLight.image2ll( make_int2(x, y) );								
		float* color = $envLight.pixel(x, y);		
		$initial[$initial_number]._weight = make_float3( color[0] * 0.6f, color[1] * 0.6f, color[2] * 0.6f );		
		$initial[$initial_number]._lambda = 1.f / 100.f;		
		writeln( $initial[$initial_number]._weight.x," ", $initial[$initial_number]._weight.y, " ", $initial[$initial_number]._weight.y );		
		$initial_number++;					
		InvalidateRect(.hwnd,NULL,0);		
		handleui();	

	OnPaint()				
		if !.hglrc:return		
		bk=.makecurrent()
		.render();
		.unmakecurrent(bk)
		.present()
		ValidateRect( .hwnd, null );

	update_texture( int l, int t, int w, int h )
		bk=.makecurrent()
		glBindTexture( GL_TEXTURE_2D, .tex );
		.pbo.bind( GL_PIXEL_UNPACK_BUFFER );
		glTexSubImage2D( GL_TEXTURE_2D, 0, l, t, w, h, .format, .type, null );		
		glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0u );
		glBindTexture( GL_TEXTURE_2D, u32(0) );
		.unmakecurrent(bk)


	render()
		glClearColor(0.f,0.f,0.5f,1.f)
		glClear( GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT );

		//draw pfm
		glBindTexture( GL_TEXTURE_2D, .tex );
		glEnable(GL_TEXTURE_2D);
		glBegin(GL_QUADS);
			glTexCoord2f(0.f, 0.f);glVertex2f(-1.f, -1.f);
			glTexCoord2f(1.f, 0.f);glVertex2f(1.f, -1.f);
			glTexCoord2f(1.f, 1.f);glVertex2f(1.f, 1.f);
			glTexCoord2f(0.f, 1.f);glVertex2f(-1.f, 1.f);
		glEnd()		
		glDisable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D,0u);

		//draw samples
		n = $initial_number;		
		if( n > 0 && $initial && .show_srbf )						
			glPushAttrib( GL_CURRENT_BIT | GL_POINT_BIT );			
			glPointSize( 8.f );
			glBegin(GL_POINTS);
			for( i=0; i<n; i++ )
				float2 point = ll2world( $initial[i]._center );								
				glColor3f( $initial[i]._lambda, 1.f, 1.f );
				glVertex2f( point.x, point.y );
			glEnd();
			glPopAttrib();	


	begin_write()
		return .pbo.map();

	end_write()
		.pbo.unmap();

	OnDestroy() PostQuitMessage(0)
	OnChar(int c)
		switch( c )
			case '\e': 
				.close();			
				break;
			case 'd':
				if $optimizing:break;
				if( $initial_number > 0 ):$initial_number--;
				InvalidateRect(.hwnd,NULL,0);
				handleui();	
				break;
			case 'e':
				writeln( "RSE: ", rse()*100.f, "%" );
				break;
			case 'r':
				if $optimizing:break;
				random_initialize(MAX_NUM_SRBF);
				InvalidateRect(.hwnd,NULL,0);
				handleui();	
				break;			
			case 'a':
				if $optimizing:break;
				random_initialize_clear($init_srbf_num);
				InvalidateRect(.hwnd,NULL,0);
				handleui();	
				break;			
			case 's':
				save_srbf( "initial.srbf", $initial, $initial_number );
				break;
			case 'o':
				$optimizing = 1;
				$loop();
				$optimizing = 0;
				break;
			case 'b':
				.show_srbf = 1 - .show_srbf;
				InvalidateRect(.hwnd,NULL,0);
				handleui();	
				break;
			case 'f':
				fill();
				InvalidateRect(.hwnd,NULL,0);
				handleui();	
				break;			
			case 'p':
				fill();
				with $envLight
					temp = new[._w*._h*._nc] float;
					cpyd2h( temp, ._approxD.d, ._w*._h*._nc*sizeof(float) );
					_output_pfm( "dump.pfm", ._w, ._h, ._nc, temp );
					delete temp;
				break;			
			case 'z':
				if $init_srbf_num>1
					$init_srbf_num--;
					writeln( "init srbf#: ", $init_srbf_num );
				break;
			case 'x':
				if $init_srbf_num<MAX_NUM_SRBF				
					$init_srbf_num++;
					writeln( "init srbf#: ", $init_srbf_num );
				break;				

window(fbWnd)	pfmWindow;
window(fbWnd)	approxWindow;

void commit_approx()
	pbo_data = $approxWindow.begin_write();
	cpyd2d( pbo_data, $envLight._approxD.d, $envLight._w * $envLight._h * $envLight._nc * sizeof(float) );
	$approxWindow.end_write();
	$approxWindow.update_texture( 0, 0, $envLight._w, $envLight._h );

	InvalidateRect($approxWindow.hwnd,NULL,0);
	InvalidateRect($pfmWindow.hwnd,NULL,0);
	handleui();		

void fill()
	n = $initial_number;		
	with $envLight		
		cuMemsetD32( (void*)(._approxD.d), __float_as_int(0.f), ._w * ._h * ._nc );
		for i=0:n-1
			srbf t = $initial[i];
			.accumulate( t );
	commit_approx();

objexport float rse()
	texture(float) approxT;
	texture(float) rasterT;
	with $envLight			
		approxT.address = ._approxD.d;
		rasterT.address = ._rasterD.d;
		f = 0.f;
		b = 0.f;
		div = 1.f/float($envLight._h * $envLight._nc);		
		spawn ._w * ._h
			tid = thread.rank;
			float2 p = .pid2ll( tid );
			if($textured)
				icoord = make_int2(tid%._w,tid/._w);
				lcoord = make_float2(lerp( 0.f, 2.f * C_PI, float(icoord.x)/float(._w) ), lerp( -C_PI/2.f, C_PI/2.f, float(icoord.y)/float(._h) ) );		
				val = get_tex_raster(lcoord);
			cos_theta = cos( p.y );
			err = 0.f;
			base = 0.f;
			float3 e;
			For i=0:2
				ii = tid * 3 + i;
				e[i] = $textured? val[i]-approxT[ii] : rasterT[ii] - approxT[ii];								
				err += e[i] * cos_theta * div * e[i];
				base += rasterT[ii] * cos_theta * div * rasterT[ii];				
			f = reduce( rop_add, err );
			b = reduce( rop_add, base );
	return f/b;

void srbf2device()
	cpyh2d( $intermediateD, $initial, $initial_number * sizeof(srbf) );
	spawn $initial_number
		tid = thread.rank;
		$initialD[tid] = $intermediateD[tid];
	

objexport int output_srbfs( char* path )
	return save_srbf( path, $initial, $initial_number );

objexport int output_dirs( char* path )
	return save_dir( path, $initial, $initial_number );

objexport int append_srbfs( void* p )
	fp = (FILE*) p;
	for( int i=0; i<$initial_number; i++ )
		fprintf( fp, "%f\t%f\t%f\t%f\t%f\t%f\t%f\n", $initial[i]._weight.x, $initial[i]._weight.y, $initial[i]._weight.z,
												$initial[i]._center.x, $initial[i]._center.y, 1.f / $initial[i]._lambda );
	return 1;	

objexport int teleport()	
	n = $initial_number;
	imin = 0;
	smin = energy($initial[imin]);	
	for i=1:n-1
		e = energy($initial[i]);
		if( e<smin )
			smin = e;
			imin = i;
	lcoord = make_float2(lerp( 0.f, 2.f * C_PI, float(rand()) / float(RAND_MAX) ), lerp( -C_PI/2.f, C_PI/2.f, float(rand()) / float(RAND_MAX) ) );		
	writeln( "teleported from #", imin, " (", $initial[imin]._center.x,",", $initial[imin]._center.y, ") to (", lcoord.x, ",", lcoord.y, ")");
	$initial[imin]._center = lcoord;
	$initial[imin]._weight = make_float3(0.f, 0.f, 0.f);
	$initial[imin]._lambda = 1.f/64.f;	
	return 1;

objexport float evaluate(double* g)			
	
	n = $initial_number;
	float* gf;
	dlist(float) gD;
	int hasG = 0;
	if( g )
		gf = new float[n * 6];
		gD = new float<n * 6>;	
		hasG = 1;
	centers_begin = 3 * n;
	lambda_begin = 5 * n;
	srbf2device();
	srbfL = $initialD;	
	
	with $envLight		
		cuMemsetD32( (void*)(._approxD.d), __float_as_int(0.f), ._w * ._h * ._nc );
		for i=0:n-1
			srbf t = $initial[i];
			.accumulate( t );					
		texture(float) approxT;
		texture(float) rasterT;
		texture(float) lambdaT;
		approxT.address = ._approxD.d;
		rasterT.address = ._rasterD.d;
		lambdaT.address = srbfL.pointerof(._lambda);
		f = 0.f;
		div = 1.f/float($envLight._h * $envLight._nc);		
		spawn ._w * ._h
			tid = thread.rank;
			if($textured)
				icoord = make_int2(tid%._w,tid/._w);
				lcoord = make_float2(lerp( 0.f, 2.f * C_PI, float(icoord.x)/float(._w) ), lerp( -C_PI/2.f, C_PI/2.f, float(icoord.y)/float(._h) ) );		
				val = get_tex_raster(lcoord);
			float2 p = .pid2ll( tid );
			cos_theta = cos( p.y );
			err = 0.f;
			float3 e;
			For i=0:2
				ii = tid * 3 + i;
				e[i] = $textured? val[i]-approxT[ii] : rasterT[ii] - approxT[ii];								
				err += e[i] * cos_theta * e[i] * div;
				e[i] *= -cos_theta * div * 2.f;
			f = reduce( rop_add, err );

		if( hasG )
			for i=0:n-1
				spawn ._w * ._h
					tid = thread.rank;
					if($textured)
						icoord = make_int2(tid%._w,tid/._w);
						lcoord = make_float2(lerp( 0.f, 2.f * C_PI, float(icoord.x)/float(._w) ), lerp( -C_PI/2.f, C_PI/2.f, float(icoord.y)/float(._h) ) );		
						val = get_tex_raster(lcoord);
					p = .pid2ll( tid );
					cos_theta = cos( p.y );
					float3 e;
					For ci=0:2
						ii = tid * 3 + ci;
						e[ci] = $textured? val[ci]-approxT[ii] : rasterT[ii] - approxT[ii];								
						e[ci] *= -cos_theta * div * 2.f;
					center = srbfL[i]._center;
					weight = srbfL[i]._weight;
					b = gdist( p, center ) - 1.f;
					c = exp( b / lambdaT[i] );
					dEdli = 0.f;
					dEdcx = 0.f;
					dEdcy = 0.f;
					For j=0:2																		//colors
						a = e[j];					
						dEdwi = a * c;					
						gD[j * n + i] = reduce( rop_add, dEdwi );
						d = dEdwi * weight[j] / lambdaT[i];
						dEdli += -d * b / lambdaT[i];
						dEdcx += d * dGDdx( center, p );
						dEdcy += d * dGDdy( center, p );
					par
						gD[lambda_begin + i] = reduce( rop_add, dEdli );
						gD[centers_begin + i] = reduce( rop_add, dEdcx );
						gD[centers_begin + n + i] = reduce( rop_add, dEdcy );
			cpyd2h( gf, gD.d, sizeof(float) * $initial_number * 6 );
			for i=0:$initial_number * 6 - 1
				g[i] = double(gf[i]/float($envLight._w));

	commit_approx();	

	return f/float($envLight._w);
	

objexport void* get_srbfs() = $initial;

objexport int get_srbf_number() = $initial_number;

objexport void cleanup()	
	if( $initialD )
		$initialD.discard();
	if( $initial )
		delete $initial;
		$initial = NULL;
	if( $intermediateD )
		ddelete $intermediateD;
		$intermediateD = NULL;
	return;

objexport int init_pfm(char* path, char* init_fn, FittingProc main_loop)			
	$envLight = new pfm;
	if ( !load_pfm( path, $envLight, $textured ) )	
		writeln( "failed in reading ", path );
		return -1;
	else
		writeln( "successfully read ", path );	

	w = new window(fbWnd);	

	with $pfmWindow = w		
		.style = WS_SYSMENU;		
		.setup( path, $envLight._w, $envLight._h, $envLight._nc, 4, 1 );				
		.create();
	pbo_data = (float*)$pfmWindow.begin_write();	
	width = $envLight._w;
	height = $envLight._h;
	nc = $envLight._nc;
	if $textured:
		spawn $envLight._w*$envLight._h
			tid=thread.rank;			
			icoord=make_int2(tid%width,tid/width);
			lcoord=make_float2( lerp( 0.f, 360.f, float(icoord.x)/float(width) ), lerp( -90.f, 90.f, float(icoord.y)/float(height) ) ) * C_D2R;						
			color=get_tex_raster(lcoord);						
			for c=0:nc-1
				((float*)pbo_data)[nc*tid+c]=color[c];		
	else
		writeln( $envLight._w * $envLight._h * $envLight._nc * sizeof(float)," bytes copied to pbo(", pbo_data, ")" )
		cpyh2d( pbo_data, $envLight._raster, $envLight._w * $envLight._h * $envLight._nc * sizeof(float) );
	$pfmWindow.end_write();	
	$pfmWindow.update_texture( 0, 0, $envLight._w, $envLight._h );

	w = new window(fbWnd);
	with $approxWindow = w
		.style = WS_SYSMENU;
		.setup( "approximation", $envLight._w, $envLight._h, $envLight._nc, 4, 1 );		
		.create();
	
	$initial = new[MAX_NUM_SRBF] srbf;	
	$intermediateD = dnew[MAX_NUM_SRBF] srbf;
	$initialD = new srbf<MAX_NUM_SRBF>;	
	if( init_fn!=NULL )
		if !read_srbf( init_fn, $initial, $initial_number )
			writeln( "failed in reading ", init_fn );
			return -2;	

	InvalidateRect($pfmWindow.hwnd,NULL,0);
	handleui();		
		
	$loop = main_loop;
	idle_msgloop($pfmWindow);	
	return 0;

objexport int update_envlight(char* name)	
	char[255] identifier;

	writeln( "update_envlight ", name );	
	f =	fopen( name, "rb" );
	if( !f )
		return -1;				
	fscanf( f, "%s\n", identifier, 255 );		
	writeln( identifier );
	nc=-1;
	if( strcmp( identifier, "Pf" ) == 0 ): nc = 1;
	else if( strcmp( identifier, "PF" ) == 0 ): nc = 3;
	else if( strcmp( identifier, "P4" ) == 0 ): nc = 4;
	else
		return -2;
	w=-1;
	h=-1;
	if( fscanf( f, "%d %d\n", &w, &h ) != 2 )		
		return -2;
	if(w!=$texwidth||h!=$texheight||nc!=$envLight._nc)
		return -3;	
	readln( f );

	writeln( w, " ", h, " ", nc );

	int nitems = w * h;
	if( fread( $envLight._raster, nc * sizeof(float), w * h, f ) != nitems )				
		return -4;
	
	writeln( "h2d" );

	cpyh2d( $envLight._rasterD.d, $envLight._raster, w * h * nc * sizeof(float) );	

	writeln( "update texture" );
			
	fth = new[w*h] float4;
	cuMemsetD32(fth,0,w*h*4);		
	for i=0:w*h-1				
		p=&$envLight._raster[i*nc];				
		for c=0:nc-1
			fth[i][c]=p[c];		
	cpyh2a($envLight._ta,fth);	
	$tex.array=$envLight._ta;
	delete fth;

	writeln( "update display" );

	pbo_data = $pfmWindow.begin_write();	
	width = $envLight._w;
	height = $envLight._h;
	nc = $envLight._nc;
	if $textured:
		spawn $envLight._w*$envLight._h
			tid=thread.rank;			
			icoord=make_int2(tid%width,tid/width);
			lcoord=make_float2( lerp( 0.f, 360.f, float(icoord.x)/float(width) ), lerp( -90.f, 90.f, float(icoord.y)/float(height) ) ) * C_D2R;						
			color=get_tex_raster(lcoord);						
			for c=0:nc-1
				((float*)pbo_data)[nc*tid+c]=color[c];		
	else		
		cpyh2d( pbo_data, $envLight._raster, $envLight._w * $envLight._h * $envLight._nc * sizeof(float) );
	$pfmWindow.end_write();	
	$pfmWindow.update_texture( 0, 0, $envLight._w, $envLight._h );

void _output_pfm( char* path, int w, int h, int nc, float* raster )	
	f = fopen( path, "wb" );
	assert( f );
	char* identifier = (nc==1)? "Pf" : "PF";
	fprintf( f, "%s\n%d %d\n%f\n", identifier, w, h, -1.f );
	fwrite( raster, nc * sizeof(float), w * h, f );
	fclose( f );