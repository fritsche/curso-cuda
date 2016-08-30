// wb.h
// by w.m.n.zola

// version 0.2
// (under construction)
//
// function prototypes / stubbs provided bellow

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define DEBUG 1
#if DEBUG == 1
   void show_debug( const char *msg ) { printf( "%s\n", msg ); }
#else
   #define show_debug( msg ) // do nothing
#endif

int Generic = 1;
int TRACE = 2;
int GPU = 3;
int Compute = 4;
int Copy = 5;

typedef struct {
   int argc;
   char **argv;
} wbArg_t;



wbArg_t wbArg_read( int argc, char **argv )
{
    wbArg_t args;
    args.argc = argc;
    args.argv = argv;

    return args;
}

void show_args( wbArg_t args )
{
    for( int i=0; i<args.argc; i++ )
      printf( "args[%d]=\"%s\"\n", i, args.argv[i] ); 
}

typedef struct {
// to do ...
} wbImage_t;

////////////// timer functions ///////////////
void wbTime_start( int timer_number, const char *message )
{
   show_debug( "wbTime_start:" ); show_debug( message );
}

void  wbTime_stop( int timer_number, const char *message )
{
}

////////////// file read functions //////////

char *wbArg_getInputFileName( wbArg_t args, int arg_number )
{
   #if DEBUG == 1
     printf( "inputfile is \"%s\"\n", args.argv[ arg_number ]  );
   #endif
   return args.argv[ arg_number ];
}


float *wbImportVector( const char *fileName, int *inputLength )
{
   int i, n;
   FILE *vectorFile;
   float *vector;

   #if DEBUG == 1
     printf( "wbImportVector fileName is \"%s\"\n", fileName );
   #endif
 
   if( strcmp( fileName, "" ) == 0 ) {
       *inputLength = 0;
       fprintf( stderr, 
         "input vector file name is \"\" (empty string)\n" );
       exit( -1 ); //return NULL;
   }

   vectorFile = fopen( fileName, "r" );
   printf( "here\n" );
   if( vectorFile == NULL ) { 
     fprintf( stderr, 
        "can\'t open input vector file \"%s\"\n", fileName );
     *inputLength = 0;
     exit( -1 ); //return NULL;
   }

   if( fscanf( vectorFile, "%d", &n ) == EOF ) {
      fprintf( stderr, 
        "premature EOF in input vector file\n" );
       *inputLength = 0;
       exit( -1 ); //return NULL;
   }

   vector = (float *)malloc( n * sizeof(float) ); 
   if( vector == NULL ) {
      fprintf( stderr, 
        "can\'t allocate host memory for input vector\n" );
      *inputLength = 0;
      exit( -1 ); //return NULL;
   }

   for( i=0; i<n; i++ ) {
      if( fscanf( vectorFile, "%f", &vector[i] ) == EOF ) {
        fprintf( stderr, 
        "only got %d floats in input vector file %s\n", i+1, fileName );
       *inputLength = 0;
       exit( -1 ); // return NULL;
      }
   }

   *inputLength = n;
   return vector; 
}

wbImage_t wbImportImage( char *fileName )
{
        fprintf( stderr, 
        "function wbImportImage to be implemented\n" );
}

////////////// logging /////////////////////
void wbLog( int log_number, const char *message, int inputLength )
{
    #if DEBUG == 1
       if( log_number == TRACE )
          printf( "TRACE[%d]: %s %d\n", log_number, message, inputLength );
    #endif
}

///////////// to present/check final solutions ///////////
void wbSolution( wbArg_t args, float *hostOutput, int inputLength )
{
        fprintf( stderr, 
        "function wbSolution for vector output to be implemented\n" );
}

 
void  wbSolution( wbArg_t args, wbImage_t outputImage )
{
        fprintf( stderr, 
        "function wbSolution for outpu images to be implemented\n" );
}

