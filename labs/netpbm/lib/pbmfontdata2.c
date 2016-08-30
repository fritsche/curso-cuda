#include "pbmfont.h"

/* Default proportional font.
   BDF-style advance value, bounding box dimensions and point of origin.


   Note: In the current version default_bdffont is fixed data, which
   means it is always accessible, without any preparatory action.
   However, the storage format may change in future versions.

   To ensure future compatibility call pbm_defaultfont() before accessing
   default_bdffont.

   The Netpbm development tool 'genfontc' generates C source code like this
   from a libnetpbm font file or builtin font.
*/

static struct glyph glBdf[190] = {
/*  32 character    */
{ 1, 1, 0, 0, 3, "\0" },
/*  33 character !  */
{ 1, 9, 1, 0, 3, "\1\1\1\1\1\1\1\0\1" },
/*  34 character "  */
{ 3, 3, 1, 6, 5, "\1\0\1\1\0\1\1\0\1" },
/*  35 character #  */
{ 5, 8, 0, 0, 6, "\0\1\0\1\0\0\1\0\1\0\1\1\1\1\1\0\1\0\1\0\0\1\0\1\0\1\1\1\1\1\0\1\0\1\0\0\1\0\1\0" },
/*  36 character $  */
{ 5, 11, 0, -1, 6, "\0\0\1\0\0\0\1\1\1\0\1\0\1\0\1\1\0\1\0\0\0\1\1\0\0\0\0\1\1\0\0\0\1\0\1\0\0\1\0\1\1\0\1\0\1\0\1\1\1\0\0\0\1\0\0" },
/*  37 character %  */
{ 8, 9, 0, 0, 9, "\0\1\1\0\0\0\1\1\1\0\0\1\1\1\1\0\1\0\0\1\0\1\0\0\0\1\1\0\1\0\0\0\0\0\0\1\0\0\0\0\0\0\1\1\0\1\1\0\0\0\1\0\1\0\0\1\0\1\0\0\1\0\0\1\0\1\0\0\0\1\1\0" },
/*  38 character &  */
{ 9, 9, 0, 0, 10, "\0\0\0\1\1\0\0\0\0\0\0\1\0\0\1\0\0\0\0\0\1\0\0\1\0\0\0\0\0\0\1\1\0\1\1\1\0\1\1\1\1\0\0\1\0\1\1\0\0\1\1\1\0\0\1\0\0\0\0\1\0\0\0\1\1\0\0\1\1\1\0\1\0\1\1\1\1\0\1\1\0" },
/*  39 character '  */
{ 2, 3, 1, 6, 4, "\1\1\0\1\1\0" },
/*  40 character (  */
{ 3, 12, 1, -3, 5, "\0\0\1\0\1\0\0\1\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\0\1\0\0\1\0\0\0\1" },
/*  41 character )  */
{ 3, 12, 0, -3, 5, "\1\0\0\0\1\0\0\1\0\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\1\0\0\1\0\1\0\0" },
/*  42 character *  */
{ 5, 5, 0, 4, 6, "\0\0\1\0\0\1\0\1\0\1\0\1\1\1\0\1\0\1\0\1\0\0\1\0\0" },
/*  43 character +  */
{ 5, 5, 1, 1, 7, "\0\0\1\0\0\0\0\1\0\0\1\1\1\1\1\0\0\1\0\0\0\0\1\0\0" },
/*  44 character ,  */
{ 2, 3, 0, -2, 3, "\0\1\0\1\1\0" },
/*  45 character -  */
{ 5, 1, 1, 3, 8, "\1\1\1\1\1" },
/*  46 character .  */
{ 1, 1, 1, 0, 3, "\1" },
/*  47 character /  */
{ 3, 9, 0, 0, 3, "\0\0\1\0\0\1\0\0\1\0\1\0\0\1\0\0\1\0\1\0\0\1\0\0\1\0\0" },
/*  48 character 0  */
{ 5, 9, 0, 0, 6, "\0\1\1\1\0\1\1\0\1\1\1\0\0\0\1\1\0\0\0\1\1\0\0\0\1\1\0\0\0\1\1\0\0\0\1\1\1\0\1\1\0\1\1\1\0" },
/*  49 character 1  */
{ 4, 9, 0, 0, 6, "\0\0\1\0\0\1\1\0\1\0\1\0\0\0\1\0\0\0\1\0\0\0\1\0\0\0\1\0\0\0\1\0\0\1\1\1" },
/*  50 character 2  */
{ 5, 9, 0, 0, 6, "\0\1\1\1\0\1\0\0\0\1\0\0\0\0\1\0\0\0\0\1\0\0\0\1\0\0\0\1\0\0\0\1\0\0\0\1\0\0\0\1\1\1\1\1\1" },
/*  51 character 3  */
{ 5, 9, 0, 0, 6, "\0\1\1\1\0\1\0\0\0\1\0\0\0\0\1\0\0\0\1\0\0\1\1\1\0\0\0\0\0\1\0\0\0\0\1\1\0\0\0\1\0\1\1\1\0" },
/*  52 character 4  */
{ 5, 9, 0, 0, 6, "\0\0\0\1\0\0\0\1\1\0\0\0\1\1\0\0\1\0\1\0\0\1\0\1\0\1\0\0\1\0\1\1\1\1\1\0\0\0\1\0\0\0\0\1\0" },
/*  53 character 5  */
{ 5, 9, 0, 0, 6, "\0\0\1\1\1\0\1\0\0\0\0\1\0\0\0\0\1\1\1\0\0\0\0\1\1\0\0\0\0\1\0\0\0\0\1\1\0\0\1\1\0\1\1\1\0" },
/*  54 character 6  */
{ 5, 9, 0, 0, 6, "\0\0\0\1\1\0\1\1\0\0\0\1\0\0\0\1\1\1\1\0\1\0\0\1\1\1\0\0\0\1\1\0\0\0\1\1\1\0\0\1\0\1\1\1\0" },
/*  55 character 7  */
{ 5, 9, 0, 0, 6, "\1\1\1\1\1\1\0\0\0\1\0\0\0\1\0\0\0\0\1\0\0\0\1\0\0\0\0\1\0\0\0\1\0\0\0\0\1\0\0\0\0\1\0\0\0" },
/*  56 character 8  */
{ 5, 9, 0, 0, 6, "\0\1\1\1\0\1\0\0\0\1\1\0\0\0\1\1\1\0\0\1\0\1\1\1\0\1\0\0\1\1\1\0\0\0\1\1\0\0\0\1\0\1\1\1\0" },
/*  57 character 9  */
{ 5, 9, 0, 0, 6, "\0\1\1\1\0\1\0\0\1\1\1\0\0\0\1\1\0\0\0\1\1\1\0\0\1\0\1\1\1\1\0\0\0\1\0\0\0\1\1\0\1\1\0\0\0" },
/*  58 character :  */
{ 1, 6, 1, 0, 3, "\1\0\0\0\0\1" },
/*  59 character ;  */
{ 2, 8, 0, -2, 3, "\0\1\0\0\0\0\0\0\0\0\0\1\0\1\1\0" },
/*  60 character <  */
{ 6, 5, 0, 1, 8, "\0\0\0\0\1\1\0\0\1\1\0\0\1\1\0\0\0\0\0\0\1\1\0\0\0\0\0\0\1\1" },
/*  61 character =  */
{ 5, 3, 1, 2, 7, "\1\1\1\1\1\0\0\0\0\0\1\1\1\1\1" },
/*  62 character >  */
{ 6, 5, 1, 1, 8, "\1\1\0\0\0\0\0\0\1\1\0\0\0\0\0\0\1\1\0\0\1\1\0\0\1\1\0\0\0\0" },
/*  63 character ?  */
{ 4, 9, 0, 0, 5, "\0\1\1\0\1\0\0\1\0\0\0\1\0\0\1\0\0\1\0\0\0\1\0\0\0\1\0\0\0\0\0\0\0\1\0\0" },
/*  64 character @  */
{ 10, 11, 1, -2, 11, "\0\0\0\0\1\1\1\1\0\0\0\0\1\1\0\0\0\0\1\0\0\1\1\0\0\0\0\0\0\1\0\1\0\0\1\1\0\1\0\1\1\0\0\1\0\0\1\0\0\1\1\0\1\0\0\0\1\0\0\1\1\0\1\0\0\1\0\0\1\0\1\0\1\0\0\1\0\0\1\0\1\0\0\1\1\0\1\1\0\0\0\1\0\0\0\0\0\0\0\0\0\0\1\1\1\1\1\0\0\0" },
/*  65 character A  */
{ 9, 9, 0, 0, 9, "\0\0\0\0\1\0\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\1\0\1\0\0\0\0\0\0\1\0\1\0\0\0\0\0\1\0\0\0\1\0\0\0\0\1\1\1\1\1\0\0\0\0\1\0\0\0\1\0\0\0\1\0\0\0\0\0\1\0\1\1\1\0\0\0\1\1\1" },
/*  66 character B  */
{ 7, 9, 0, 0, 8, "\1\1\1\1\1\1\0\0\1\0\0\0\1\1\0\1\0\0\0\0\1\0\1\0\0\0\1\1\0\1\1\1\1\1\0\0\1\0\0\0\0\1\0\1\0\0\0\0\1\0\1\0\0\0\1\1\1\1\1\1\1\1\0" },
/*  67 character C  */
{ 7, 9, 0, 0, 8, "\0\0\1\1\1\0\1\0\1\1\0\0\1\1\0\1\0\0\0\0\1\1\0\0\0\0\0\0\1\0\0\0\0\0\0\1\0\0\0\0\0\0\0\1\0\0\0\0\1\0\1\1\0\0\1\1\0\0\1\1\1\1\0" },
/*  68 character D  */
{ 8, 9, 0, 0, 9, "\1\1\1\1\1\1\0\0\0\1\0\0\0\1\1\0\0\1\0\0\0\0\1\0\0\1\0\0\0\0\0\1\0\1\0\0\0\0\0\1\0\1\0\0\0\0\0\1\0\1\0\0\0\0\1\0\0\1\0\0\0\1\1\0\1\1\1\1\1\1\0\0" },
/*  69 character E  */
{ 7, 9, 0, 0, 8, "\1\1\1\1\1\1\1\0\1\0\0\0\0\1\0\1\0\0\0\0\0\0\1\0\0\0\1\0\0\1\1\1\1\1\0\0\1\0\0\0\1\0\0\1\0\0\0\0\0\0\1\0\0\0\0\1\1\1\1\1\1\1\1" },
/*  70 character F  */
{ 7, 9, 0, 0, 8, "\1\1\1\1\1\1\1\0\1\0\0\0\0\1\0\1\0\0\0\0\0\0\1\0\0\0\1\0\0\1\1\1\1\1\0\0\1\0\0\0\1\0\0\1\0\0\0\0\0\0\1\0\0\0\0\0\1\1\1\1\0\0\0" },
/*  71 character G  */
{ 8, 9, 0, 0, 9, "\0\0\1\1\1\0\1\0\0\1\1\0\0\1\1\0\0\1\0\0\0\0\1\0\1\0\0\0\0\0\0\0\1\0\0\0\0\1\1\1\1\0\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\1\1\0\0\1\1\0\0\0\1\1\1\1\0\0" },
/*  72 character H  */
{ 8, 9, 0, 0, 9, "\1\1\1\0\0\1\1\1\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\1\1\1\1\1\1\0\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\1\1\1\0\0\1\1\1" },
/*  73 character I  */
{ 3, 9, 0, 0, 4, "\1\1\1\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\1\1\1" },
/*  74 character J  */
{ 4, 9, 0, 0, 4, "\0\1\1\1\0\0\1\0\0\0\1\0\0\0\1\0\0\0\1\0\0\0\1\0\0\0\1\0\1\0\1\0\1\1\0\0" },
/*  75 character K  */
{ 8, 9, 0, 0, 8, "\1\1\1\0\1\1\1\0\0\1\0\0\0\1\0\0\0\1\0\0\1\0\0\0\0\1\0\1\0\0\0\0\0\1\1\1\0\0\0\0\0\1\0\1\1\0\0\0\0\1\0\0\1\1\0\0\0\1\0\0\0\1\1\0\1\1\1\0\0\1\1\1" },
/*  76 character L  */
{ 6, 9, 0, 0, 7, "\1\1\1\0\0\0\0\1\0\0\0\0\0\1\0\0\0\0\0\1\0\0\0\0\0\1\0\0\0\0\0\1\0\0\0\0\0\1\0\0\0\0\0\1\0\0\0\1\1\1\1\1\1\1" },
/*  77 character M  */
{ 11, 9, 0, 0, 11, "\1\1\0\0\0\0\0\0\0\1\1\0\1\1\0\0\0\0\0\1\1\0\0\1\1\0\0\0\0\0\1\1\0\0\1\0\1\0\0\0\1\0\1\0\0\1\0\1\0\0\0\1\0\1\0\0\1\0\0\1\0\1\0\0\1\0\0\1\0\0\1\0\1\0\0\1\0\0\1\0\0\0\1\0\0\0\1\0\1\1\1\0\0\1\0\0\1\1\1" },
/*  78 character N  */
{ 9, 9, 0, 0, 9, "\1\1\0\0\0\0\1\1\1\0\1\1\0\0\0\0\1\0\0\1\1\0\0\0\0\1\0\0\1\0\1\0\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\0\1\0\1\0\0\1\0\0\0\0\1\1\0\1\1\1\0\0\0\0\1\0" },
/*  79 character O  */
{ 8, 9, 0, 0, 9, "\0\0\1\1\1\1\0\0\0\1\1\0\0\1\1\0\0\1\0\0\0\0\1\0\1\0\0\0\0\0\0\1\1\0\0\0\0\0\0\1\1\0\0\0\0\0\0\1\0\1\0\0\0\0\1\0\0\1\1\0\0\1\1\0\0\0\1\1\1\1\0\0" },
/*  80 character P  */
{ 7, 9, 0, 0, 7, "\1\1\1\1\1\1\0\0\1\0\0\0\1\1\0\1\0\0\0\0\1\0\1\0\0\0\1\1\0\1\1\1\1\1\0\0\1\0\0\0\0\0\0\1\0\0\0\0\0\0\1\0\0\0\0\0\1\1\1\0\0\0\0" },
/*  81 character Q  */
{ 8, 11, 0, -2, 9, "\0\0\1\1\1\1\0\0\0\1\1\0\0\1\1\0\0\1\0\0\0\0\1\0\1\0\0\0\0\0\0\1\1\0\0\0\0\0\0\1\1\0\0\0\0\0\0\1\0\1\0\0\0\0\1\0\0\1\1\0\0\1\1\0\0\0\1\1\1\1\0\0\0\0\0\0\1\1\0\0\0\0\0\0\0\0\1\1" },
/*  82 character R  */
{ 8, 9, 0, 0, 8, "\1\1\1\1\1\1\0\0\0\1\0\0\0\1\1\0\0\1\0\0\0\0\1\0\0\1\0\0\0\1\1\0\0\1\1\1\1\1\0\0\0\1\0\0\1\0\0\0\0\1\0\0\0\1\0\0\0\1\0\0\0\0\1\0\1\1\1\0\0\0\1\1" },
/*  83 character S  */
{ 6, 9, 0, 0, 7, "\0\1\1\1\0\1\1\0\0\0\1\1\1\0\0\0\0\1\0\1\1\0\0\0\0\0\1\1\1\0\0\0\0\0\1\1\1\0\0\0\0\1\1\1\0\0\1\1\1\0\1\1\1\0" },
/*  84 character T  */
{ 7, 9, 0, 0, 7, "\1\1\1\1\1\1\1\1\0\0\1\0\0\1\0\0\0\1\0\0\0\0\0\0\1\0\0\0\0\0\0\1\0\0\0\0\0\0\1\0\0\0\0\0\0\1\0\0\0\0\0\0\1\0\0\0\0\0\1\1\1\0\0" },
/*  85 character U  */
{ 8, 9, 0, 0, 8, "\1\1\1\0\0\1\1\1\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\1\1\0\0\1\1\0\0\0\1\1\1\1\0\0" },
/*  86 character V  */
{ 9, 9, 0, 0, 9, "\1\1\1\0\0\0\1\1\1\0\1\0\0\0\0\0\1\0\0\0\1\0\0\0\1\0\0\0\0\1\0\0\0\1\0\0\0\0\0\1\0\1\0\0\0\0\0\0\1\0\1\0\0\0\0\0\0\1\1\1\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\0\1\0\0\0\0" },
/*  87 character W  */
{ 12, 9, 0, 0, 12, "\1\1\1\0\1\1\1\0\0\1\1\1\0\1\0\0\0\1\0\0\0\0\1\0\0\1\1\0\0\1\1\0\0\0\1\0\0\0\1\0\0\0\1\0\0\1\0\0\0\0\1\1\0\1\1\1\0\1\0\0\0\0\0\1\0\1\0\1\0\1\0\0\0\0\0\1\1\0\0\1\1\0\0\0\0\0\0\0\1\0\0\0\1\0\0\0\0\0\0\0\1\0\0\0\1\0\0\0" },
/*  88 character X  */
{ 8, 9, 0, 0, 8, "\1\1\1\0\0\1\1\1\0\1\0\0\0\0\1\0\0\0\1\0\0\1\0\0\0\0\1\1\1\0\0\0\0\0\0\1\1\0\0\0\0\0\1\0\1\1\0\0\0\0\1\0\0\1\0\0\0\1\0\0\0\0\1\0\1\1\1\0\0\1\1\1" },
/*  89 character Y  */
{ 9, 9, 0, 0, 9, "\1\1\1\0\0\0\1\1\1\0\1\0\0\0\0\0\1\0\0\0\1\0\0\0\1\0\0\0\0\0\1\0\0\1\0\0\0\0\0\1\1\1\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\1\1\1\0\0\0" },
/*  90 character Z  */
{ 7, 9, 0, 0, 8, "\1\1\1\1\1\1\1\1\0\0\0\0\1\1\0\0\0\0\1\1\0\0\0\0\1\1\0\0\0\0\1\1\0\0\0\0\1\1\0\0\0\0\1\1\0\0\0\0\0\1\0\0\0\0\0\1\1\1\1\1\1\1\1" },
/*  91 character [  */
{ 3, 12, 1, -3, 5, "\1\1\1\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\1\1" },
/*  92 character \  */
{ 3, 9, 0, 0, 3, "\1\0\0\1\0\0\1\0\0\0\1\0\0\1\0\0\1\0\0\0\1\0\0\1\0\0\1" },
/*  93 character ]  */
{ 3, 12, 0, -3, 5, "\1\1\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\1\1\1" },
/*  94 character ^  */
{ 5, 5, 0, 4, 6, "\0\0\1\0\0\0\1\0\1\0\0\1\0\1\0\1\0\0\0\1\1\0\0\0\1" },
/*  95 character _  */
{ 6, 1, 0, -3, 6, "\1\1\1\1\1\1" },
/*  96 character `  */
{ 2, 3, 1, 6, 4, "\0\1\1\0\1\1" },
/*  97 character a  */
{ 5, 6, 1, 0, 6, "\0\1\1\0\0\1\0\0\1\0\0\1\1\1\0\1\0\0\1\0\1\0\0\1\0\0\1\1\0\1" },
/*  98 character b  */
{ 5, 9, 0, 0, 6, "\1\1\0\0\0\0\1\0\0\0\0\1\0\0\0\0\1\1\1\0\0\1\0\0\1\0\1\0\0\1\0\1\0\0\1\0\1\0\0\1\0\1\1\1\0" },
/*  99 character c  */
{ 4, 6, 1, 0, 5, "\0\1\1\0\1\0\0\1\1\0\0\0\1\0\0\0\1\0\0\1\0\1\1\0" },
/* 100 character d  */
{ 5, 9, 1, 0, 6, "\0\0\1\1\0\0\0\0\1\0\0\0\0\1\0\0\1\1\1\0\1\0\0\1\0\1\0\0\1\0\1\0\0\1\0\1\0\0\1\0\0\1\1\0\1" },
/* 101 character e  */
{ 5, 6, 1, 0, 6, "\0\1\1\0\0\1\0\0\1\0\1\1\1\1\0\1\0\0\0\0\1\1\0\0\1\0\1\1\1\0" },
/* 102 character f  */
{ 3, 9, 0, 0, 3, "\0\0\1\0\1\0\0\1\0\1\1\1\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0" },
/* 103 character g  */
{ 5, 9, 1, -3, 6, "\0\1\1\1\1\1\0\0\1\0\1\0\0\1\0\1\1\1\0\0\0\1\0\0\0\0\1\1\1\0\1\0\0\0\1\1\0\0\0\1\0\1\1\1\0" },
/* 104 character h  */
{ 6, 9, 0, 0, 6, "\1\1\0\0\0\0\0\1\0\0\0\0\0\1\0\0\0\0\0\1\1\1\0\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\1\1\1\0\1\1" },
/* 105 character i  */
{ 3, 9, 0, 0, 3, "\0\1\0\0\0\0\0\0\0\1\1\0\0\1\0\0\1\0\0\1\0\0\1\0\1\1\1" },
/* 106 character j  */
{ 2, 12, 0, -3, 3, "\0\1\0\0\0\0\1\1\0\1\0\1\0\1\0\1\0\1\0\1\0\1\1\0" },
/* 107 character k  */
{ 6, 9, 0, 0, 6, "\1\1\0\0\0\0\0\1\0\0\0\0\0\1\0\0\0\0\0\1\0\0\1\0\0\1\0\1\0\0\0\1\1\0\0\0\0\1\0\1\0\0\0\1\0\0\1\0\0\1\0\0\1\1" },
/* 108 character l  */
{ 3, 9, 0, 0, 3, "\1\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\1\1\1" },
/* 109 character m  */
{ 9, 6, 0, 0, 9, "\1\0\1\1\0\1\1\0\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\1\1\1\0\1\1\0\1\1" },
/* 110 character n  */
{ 6, 6, 0, 0, 6, "\1\0\1\1\0\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\1\1\1\0\1\1" },
/* 111 character o  */
{ 4, 6, 1, 0, 6, "\0\1\1\0\1\0\0\1\1\0\0\1\1\0\0\1\1\0\0\1\0\1\1\0" },
/* 112 character p  */
{ 5, 9, 0, -3, 6, "\1\1\1\1\0\0\1\0\0\1\0\1\0\0\1\0\1\0\0\1\0\1\0\0\1\0\1\1\1\0\0\1\0\0\0\0\1\0\0\0\1\1\1\0\0" },
/* 113 character q  */
{ 5, 9, 1, -3, 6, "\0\1\1\1\0\1\0\0\1\0\1\0\0\1\0\1\0\0\1\0\1\0\0\1\0\0\1\1\1\0\0\0\0\1\0\0\0\0\1\0\0\0\1\1\1" },
/* 114 character r  */
{ 4, 6, 0, 0, 4, "\1\0\1\1\0\1\1\0\0\1\0\0\0\1\0\0\0\1\0\0\1\1\1\0" },
/* 115 character s  */
{ 4, 6, 1, 0, 6, "\0\1\1\1\1\0\0\1\1\1\0\0\0\0\1\1\1\0\0\1\1\1\1\0" },
/* 116 character t  */
{ 4, 7, 0, 0, 4, "\0\1\0\0\1\1\1\1\0\1\0\0\0\1\0\0\0\1\0\0\0\1\0\0\0\0\1\1" },
/* 117 character u  */
{ 6, 6, 0, 0, 6, "\1\1\0\1\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\0\1\1\0\1" },
/* 118 character v  */
{ 6, 6, 0, 0, 6, "\1\1\0\0\1\1\0\1\0\0\1\0\0\1\0\1\1\0\0\1\0\1\0\0\0\0\1\1\0\0\0\0\1\0\0\0" },
/* 119 character w  */
{ 9, 6, 0, 0, 9, "\1\1\1\0\1\1\0\1\1\0\1\0\0\1\0\0\1\0\0\1\1\0\1\0\1\1\0\0\0\1\0\1\0\1\0\0\0\0\1\1\0\1\0\0\0\0\0\1\0\0\1\0\0\0" },
/* 120 character x  */
{ 5, 6, 1, 0, 6, "\1\1\0\1\1\0\1\0\1\0\0\0\1\0\0\0\0\1\0\0\0\1\0\1\0\1\1\0\1\1" },
/* 121 character y  */
{ 6, 9, 0, -3, 6, "\1\1\0\0\1\1\0\1\0\0\1\0\0\1\0\1\1\0\0\1\0\1\0\0\0\0\1\1\0\0\0\0\1\0\0\0\0\0\1\0\0\0\0\1\0\0\0\0\1\1\0\0\0\0" },
/* 122 character z  */
{ 4, 6, 1, 0, 6, "\1\1\1\1\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\1\1\1\1" },
/* 123 character {  */
{ 4, 12, 1, -3, 6, "\0\0\1\1\0\1\0\0\0\1\0\0\0\1\0\0\0\1\0\0\1\0\0\0\0\1\0\0\0\1\0\0\0\1\0\0\0\1\0\0\0\1\0\0\0\0\1\1" },
/* 124 character |  */
{ 1, 9, 1, 0, 3, "\1\1\1\1\1\1\1\1\1" },
/* 125 character }  */
{ 4, 12, 0, -3, 6, "\1\1\0\0\0\0\1\0\0\0\1\0\0\0\1\0\0\0\1\0\0\0\0\1\0\0\1\0\0\0\1\0\0\0\1\0\0\0\1\0\0\0\1\0\1\1\0\0" },
/* 160 */
{ 6, 2, 0, 3, 7, "\0\1\1\0\0\1\1\0\0\1\1\0" },
/* 161 */
{ 1, 9, 1, -3, 4, "\1\0\1\1\1\1\1\1\1" },
/* 162 */
{ 5, 8, 1, -1, 6, "\0\0\0\0\1\0\1\1\1\0\1\0\0\1\1\1\0\1\0\0\1\0\1\0\0\1\1\0\0\1\0\1\1\1\0\1\0\0\0\0" },
/* 163 */
{ 5, 9, 0, 0, 6, "\0\0\1\1\0\0\1\0\0\1\0\1\0\0\0\0\1\0\0\0\1\1\1\1\0\0\1\0\0\0\0\1\0\0\0\1\1\1\0\1\1\1\0\1\1" },
/* 164 */
{ 6, 7, 1, 1, 7, "\1\0\0\0\0\1\0\1\1\1\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\1\1\1\0\1\0\0\0\0\1" },
/* 165 */
{ 5, 9, 0, 0, 6, "\1\0\0\0\1\1\0\0\0\1\0\1\0\1\0\0\1\0\1\0\1\1\1\1\1\0\0\1\0\0\1\1\1\1\1\0\0\1\0\0\0\1\1\1\0" },
/* 166 */
{ 1, 9, 1, 0, 3, "\1\1\1\0\0\1\1\1\1" },
/* 167 */
{ 4, 12, 1, -3, 6, "\0\1\1\1\1\0\0\1\1\1\0\0\0\1\1\0\1\0\1\1\1\0\0\1\1\0\0\1\1\1\0\1\0\1\1\0\0\0\1\1\1\0\0\1\1\1\1\0" },
/* 168 */
{ 3, 1, 0, 7, 3, "\1\0\1" },
/* 169 */
{ 9, 9, 1, 0, 11, "\0\0\0\1\1\1\0\0\0\0\1\1\0\0\0\1\1\0\0\1\0\1\1\1\0\1\0\1\0\1\0\0\1\0\0\1\1\0\1\0\0\0\0\0\1\1\0\1\0\0\1\0\0\1\0\1\0\1\1\1\0\1\0\0\1\1\0\0\0\1\1\0\0\0\0\1\1\1\0\0\0" },
/* 170 */
{ 3, 6, 1, 3, 5, "\1\1\0\0\0\1\1\1\1\1\0\1\0\0\0\1\1\1" },
/* 171 */
{ 5, 5, 1, 0, 7, "\0\0\1\0\1\0\1\0\1\0\1\0\1\0\0\0\1\0\1\0\0\0\1\0\1" },
/* 172 */
{ 6, 4, 1, 1, 8, "\1\1\1\1\1\1\0\0\0\0\0\1\0\0\0\0\0\1\0\0\0\0\0\1" },
/* 173 */
{ 4, 1, 1, 3, 6, "\1\1\1\1" },
/* 174 */
{ 9, 9, 1, 0, 11, "\0\0\0\1\1\1\0\0\0\0\1\1\0\0\0\1\1\0\0\1\0\1\1\1\0\1\0\1\0\0\1\0\0\1\0\1\1\0\0\1\1\1\0\0\1\1\0\0\1\0\1\0\0\1\1\1\0\1\0\1\0\1\0\0\1\1\0\0\0\1\1\0\0\0\1\1\1\1\0\0\0" },
/* 175 */
{ 4, 1, 0, 7, 4, "\1\1\1\1" },
/* 176 */
{ 4, 4, 0, 5, 5, "\0\1\1\0\1\0\0\1\1\0\0\1\0\1\1\0" },
/* 177 */
{ 5, 7, 1, 0, 7, "\0\0\1\0\0\0\0\1\0\0\1\1\1\1\1\0\0\1\0\0\0\0\1\0\0\0\0\0\0\0\1\1\1\1\1" },
/* 178 */
{ 4, 5, 0, 4, 4, "\0\1\1\0\1\0\0\1\0\0\1\0\0\1\0\0\1\1\1\1" },
/* 179 */
{ 3, 5, 0, 4, 4, "\1\1\1\0\0\1\0\1\0\0\0\1\1\1\0" },
/* 180 */
{ 2, 2, 1, 7, 4, "\0\1\1\0" },
/* 181 */
{ 6, 9, 0, -3, 6, "\1\1\0\1\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\1\1\0\1\0\1\0\0\0\0\0\1\0\0\0\0\0\1\1\0\0\0" },
/* 182 */
{ 6, 12, 0, -3, 7, "\0\1\1\1\1\1\1\1\1\0\1\0\1\1\1\0\1\0\1\1\1\0\1\0\1\1\1\0\1\0\0\1\1\0\1\0\0\0\1\0\1\0\0\0\1\0\1\0\0\0\1\0\1\0\0\0\1\0\1\0\0\0\1\0\1\0\0\0\1\0\1\0" },
/* 183 */
{ 1, 1, 1, 3, 3, "\1" },
/* 184 */
{ 3, 3, 0, -3, 3, "\0\1\0\0\0\1\1\1\1" },
/* 185 */
{ 3, 5, 0, 4, 4, "\0\1\0\1\1\0\0\1\0\0\1\0\1\1\1" },
/* 186 */
{ 3, 6, 1, 3, 5, "\0\1\0\1\0\1\1\0\1\0\1\0\0\0\0\1\1\1" },
/* 187 */
{ 5, 5, 0, 0, 7, "\1\0\1\0\0\0\1\0\1\0\0\0\1\0\1\0\1\0\1\0\1\0\1\0\0" },
/* 188 */
{ 9, 9, 0, 0, 9, "\0\1\0\0\0\0\1\0\0\1\1\0\0\0\1\0\0\0\0\1\0\0\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\1\1\0\1\0\0\1\0\0\0\0\1\0\0\1\1\0\0\0\0\1\0\1\0\1\0\0\0\1\0\0\1\1\1\1\0\0\1\0\0\0\0\1\0" },
/* 189 */
{ 9, 9, 0, 0, 9, "\0\1\0\0\0\0\1\0\0\1\1\0\0\0\1\0\0\0\0\1\0\0\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\1\1\0\1\0\1\1\0\0\0\0\1\0\1\0\0\1\0\0\0\1\0\0\0\1\0\0\0\1\0\0\0\1\0\0\0\0\1\0\0\1\1\1\1" },
/* 190 */
{ 9, 9, 0, 0, 9, "\1\1\1\0\0\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\0\1\0\0\0\0\0\1\0\1\0\0\0\0\1\1\0\0\1\0\0\1\0\0\0\0\1\0\0\1\1\0\0\0\0\1\0\1\0\1\0\0\0\1\0\0\1\1\1\1\0\0\1\0\0\0\0\1\0" },
/* 191 */
{ 4, 9, 0, -3, 5, "\0\0\1\0\0\0\0\0\0\0\1\0\0\0\1\0\0\0\1\0\0\1\0\0\1\0\0\0\1\0\0\1\0\1\1\0" },
/* 192 */
{ 9, 12, 0, 0, 9, "\0\0\0\1\0\0\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\1\0\1\0\0\0\0\0\0\1\0\1\0\0\0\0\0\1\0\0\0\1\0\0\0\0\1\1\1\1\1\0\0\0\0\1\0\0\0\1\0\0\0\1\0\0\0\0\0\1\0\1\1\1\0\0\0\1\1\1" },
/* 193 */
{ 9, 12, 0, 0, 9, "\0\0\0\0\0\1\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\1\0\1\0\0\0\0\0\0\1\0\1\0\0\0\0\0\1\0\0\0\1\0\0\0\0\1\1\1\1\1\0\0\0\0\1\0\0\0\1\0\0\0\1\0\0\0\0\0\1\0\1\1\1\0\0\0\1\1\1" },
/* 194 */
{ 9, 12, 0, 0, 9, "\0\0\0\0\1\0\0\0\0\0\0\0\1\0\1\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\1\0\1\0\0\0\0\0\0\1\0\1\0\0\0\0\0\1\0\0\0\1\0\0\0\0\1\1\1\1\1\0\0\0\0\1\0\0\0\1\0\0\0\1\0\0\0\0\0\1\0\1\1\1\0\0\0\1\1\1" },
/* 195 */
{ 9, 12, 0, 0, 9, "\0\0\0\0\1\0\1\0\0\0\0\0\1\0\1\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\1\0\1\0\0\0\0\0\0\1\0\1\0\0\0\0\0\1\0\0\0\1\0\0\0\0\1\1\1\1\1\0\0\0\0\1\0\0\0\1\0\0\0\1\0\0\0\0\0\1\0\1\1\1\0\0\0\1\1\1" },
/* 196 */
{ 9, 11, 0, 0, 9, "\0\0\0\1\0\1\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\1\0\1\0\0\0\0\0\0\1\0\1\0\0\0\0\0\1\0\0\0\1\0\0\0\0\1\1\1\1\1\0\0\0\0\1\0\0\0\1\0\0\0\1\0\0\0\0\0\1\0\1\1\1\0\0\0\1\1\1" },
/* 197 */
{ 9, 12, 0, 0, 9, "\0\0\0\0\1\0\0\0\0\0\0\0\1\0\1\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\1\0\1\0\0\0\0\0\0\1\0\1\0\0\0\0\0\1\0\0\0\1\0\0\0\0\1\1\1\1\1\0\0\0\0\1\0\0\0\1\0\0\0\1\0\0\0\0\0\1\0\1\1\1\0\0\0\1\1\1" },
/* 198 */
{ 10, 9, 0, 0, 11, "\0\0\1\1\1\1\1\1\1\1\0\0\0\1\1\0\0\0\0\1\0\0\1\0\1\0\0\0\0\0\0\0\1\0\1\0\0\0\1\0\0\1\0\0\1\1\1\1\1\0\0\1\1\1\1\0\0\0\1\0\0\1\0\0\1\0\0\0\0\0\1\0\0\0\1\0\0\0\0\1\1\1\0\0\1\1\1\1\1\1" },
/* 199 */
{ 7, 12, 0, -3, 8, "\0\0\1\1\1\0\1\0\1\1\0\0\1\1\0\1\0\0\0\0\1\1\0\0\0\0\0\0\1\0\0\0\0\0\0\1\0\0\0\0\0\0\0\1\0\0\0\0\1\0\1\1\0\0\1\1\0\0\1\1\1\1\0\0\0\0\1\0\0\0\0\0\0\0\1\0\0\0\0\1\1\1\0\0" },
/* 200 */
{ 7, 12, 0, 0, 8, "\0\0\1\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\0\0\0\1\1\1\1\1\1\1\0\1\0\0\0\0\1\0\1\0\0\0\0\0\0\1\0\0\0\1\0\0\1\1\1\1\1\0\0\1\0\0\0\1\0\0\1\0\0\0\0\0\0\1\0\0\0\0\1\1\1\1\1\1\1\1" },
/* 201 */
{ 7, 12, 0, 0, 8, "\0\0\0\0\1\0\0\0\0\0\1\0\0\0\0\0\0\0\0\0\0\1\1\1\1\1\1\1\0\1\0\0\0\0\1\0\1\0\0\0\0\0\0\1\0\0\0\1\0\0\1\1\1\1\1\0\0\1\0\0\0\1\0\0\1\0\0\0\0\0\0\1\0\0\0\0\1\1\1\1\1\1\1\1" },
/* 202 */
{ 7, 12, 0, 0, 8, "\0\0\0\1\0\0\0\0\0\1\0\1\0\0\0\0\0\0\0\0\0\1\1\1\1\1\1\1\0\1\0\0\0\0\1\0\1\0\0\0\0\0\0\1\0\0\0\1\0\0\1\1\1\1\1\0\0\1\0\0\0\1\0\0\1\0\0\0\0\0\0\1\0\0\0\0\1\1\1\1\1\1\1\1" },
/* 203 */
{ 7, 11, 0, 0, 8, "\0\0\1\0\1\0\0\0\0\0\0\0\0\0\1\1\1\1\1\1\1\0\1\0\0\0\0\1\0\1\0\0\0\0\0\0\1\0\0\0\1\0\0\1\1\1\1\1\0\0\1\0\0\0\1\0\0\1\0\0\0\0\0\0\1\0\0\0\0\1\1\1\1\1\1\1\1" },
/* 204 */
{ 3, 12, 0, 0, 4, "\1\0\0\0\1\0\0\0\0\1\1\1\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\1\1\1" },
/* 205 */
{ 3, 12, 0, 0, 4, "\0\0\1\0\1\0\0\0\0\1\1\1\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\1\1\1" },
/* 206 */
{ 3, 12, 0, 0, 4, "\0\1\0\1\0\1\0\0\0\1\1\1\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\1\1\1" },
/* 207 */
{ 3, 11, 0, 0, 4, "\1\0\1\0\0\0\1\1\1\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\1\1\1" },
/* 208 */
{ 8, 9, 0, 0, 9, "\1\1\1\1\1\1\0\0\0\1\0\0\0\1\1\0\0\1\0\0\0\0\1\0\0\1\0\0\0\0\0\1\1\1\1\0\0\0\0\1\0\1\0\0\0\0\0\1\0\1\0\0\0\0\1\0\0\1\0\0\0\1\1\0\1\1\1\1\1\1\0\0" },
/* 209 */
{ 9, 12, 0, 0, 9, "\0\0\0\0\1\0\1\0\0\0\0\0\1\0\1\0\0\0\0\0\0\0\0\0\0\0\0\1\1\0\0\0\0\1\1\1\0\1\1\0\0\0\0\1\0\0\1\1\0\0\0\0\1\0\0\1\0\1\0\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\0\1\0\1\0\0\1\0\0\0\0\1\1\0\1\1\1\0\0\0\0\1\0" },
/* 210 */
{ 8, 12, 0, 0, 9, "\0\0\1\0\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\0\0\0\0\0\0\0\1\1\1\1\0\0\0\1\1\0\0\1\1\0\0\1\0\0\0\0\1\0\1\0\0\0\0\0\0\1\1\0\0\0\0\0\0\1\1\0\0\0\0\0\0\1\0\1\0\0\0\0\1\0\0\1\1\0\0\1\1\0\0\0\1\1\1\1\0\0" },
/* 211 */
{ 8, 12, 0, 0, 9, "\0\0\0\0\0\1\0\0\0\0\0\0\1\0\0\0\0\0\0\0\0\0\0\0\0\0\1\1\1\1\0\0\0\1\1\0\0\1\1\0\0\1\0\0\0\0\1\0\1\0\0\0\0\0\0\1\1\0\0\0\0\0\0\1\1\0\0\0\0\0\0\1\0\1\0\0\0\0\1\0\0\1\1\0\0\1\1\0\0\0\1\1\1\1\0\0" },
/* 212 */
{ 8, 12, 0, 0, 9, "\0\0\0\1\0\0\0\0\0\0\1\0\1\0\0\0\0\0\0\0\0\0\0\0\0\0\1\1\1\1\0\0\0\1\1\0\0\1\1\0\0\1\0\0\0\0\1\0\1\0\0\0\0\0\0\1\1\0\0\0\0\0\0\1\1\0\0\0\0\0\0\1\0\1\0\0\0\0\1\0\0\1\1\0\0\1\1\0\0\0\1\1\1\1\0\0" },
/* 213 */
{ 8, 12, 0, 0, 9, "\0\0\0\1\0\1\0\0\0\0\1\0\1\0\0\0\0\0\0\0\0\0\0\0\0\0\1\1\1\1\0\0\0\1\1\0\0\1\1\0\0\1\0\0\0\0\1\0\1\0\0\0\0\0\0\1\1\0\0\0\0\0\0\1\1\0\0\0\0\0\0\1\0\1\0\0\0\0\1\0\0\1\1\0\0\1\1\0\0\0\1\1\1\1\0\0" },
/* 214 */
{ 8, 11, 0, 0, 9, "\0\0\1\0\0\1\0\0\0\0\0\0\0\0\0\0\0\0\1\1\1\1\0\0\0\1\1\0\0\1\1\0\0\1\0\0\0\0\1\0\1\0\0\0\0\0\0\1\1\0\0\0\0\0\0\1\1\0\0\0\0\0\0\1\0\1\0\0\0\0\1\0\0\1\1\0\0\1\1\0\0\0\1\1\1\1\0\0" },
/* 215 */
{ 5, 5, 1, 1, 7, "\1\0\0\0\1\0\1\0\1\0\0\0\1\0\0\0\1\0\1\0\1\0\0\0\1" },
/* 216 */
{ 9, 10, 0, 0, 9, "\0\0\0\0\0\0\0\0\1\0\0\1\1\1\1\0\1\0\0\1\1\0\0\1\1\0\0\0\1\0\0\0\1\1\0\0\1\0\0\0\1\0\0\1\0\1\0\0\0\1\0\0\1\0\1\0\0\1\0\0\0\1\0\0\1\1\0\0\0\1\0\0\0\1\1\0\0\1\1\0\0\1\0\1\1\1\1\0\0\0" },
/* 217 */
{ 8, 12, 0, 0, 8, "\0\0\0\1\0\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\0\0\0\0\1\1\1\0\0\1\1\1\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\1\1\0\0\1\1\0\0\0\1\1\1\1\0\0" },
/* 218 */
{ 8, 12, 0, 0, 8, "\0\0\0\0\0\1\0\0\0\0\0\0\1\0\0\0\0\0\0\0\0\0\0\0\1\1\1\0\0\1\1\1\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\1\1\0\0\1\1\0\0\0\1\1\1\1\0\0" },
/* 219 */
{ 8, 12, 0, 0, 8, "\0\0\0\1\0\0\0\0\0\0\1\0\1\0\0\0\0\0\0\0\0\0\0\0\1\1\1\0\0\1\1\1\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\1\1\0\0\1\1\0\0\0\1\1\1\1\0\0" },
/* 220 */
{ 8, 11, 0, 0, 8, "\0\0\1\0\1\0\0\0\0\0\0\0\0\0\0\0\1\1\1\0\0\1\1\1\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\1\0\0\0\0\1\0\0\1\1\0\0\1\1\0\0\0\1\1\1\1\0\0" },
/* 221 */
{ 9, 12, 0, 0, 9, "\0\0\0\0\0\1\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\0\0\0\0\0\0\1\1\1\0\0\0\1\1\1\0\1\0\0\0\0\0\1\0\0\0\1\0\0\0\1\0\0\0\0\0\1\0\0\1\0\0\0\0\0\1\1\1\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\1\1\1\0\0\0" },
/* 222 */
{ 7, 9, 0, 0, 7, "\1\1\1\0\0\0\0\0\1\0\0\0\0\0\0\1\1\1\1\1\0\0\1\0\0\0\1\1\0\1\0\0\0\0\1\0\1\0\0\0\1\1\0\1\1\1\1\1\0\0\1\0\0\0\0\0\1\1\1\0\0\0\0" },
/* 223 */
{ 6, 9, 0, 0, 6, "\0\0\1\1\0\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\1\0\0\0\1\1\1\0\0\0\1\0\0\1\0\0\1\0\0\0\1\0\1\0\0\0\1\1\1\0\1\1\0" },
/* 224 */
{ 5, 9, 1, 0, 6, "\0\1\0\0\0\0\0\1\0\0\0\0\0\0\0\0\1\1\0\0\1\0\0\1\0\0\1\1\1\0\1\0\0\1\0\1\0\0\1\0\0\1\1\0\1" },
/* 225 */
{ 5, 9, 1, 0, 6, "\0\0\0\1\0\0\0\1\0\0\0\0\0\0\0\0\1\1\0\0\1\0\0\1\0\0\1\1\1\0\1\0\0\1\0\1\0\0\1\0\0\1\1\0\1" },
/* 226 */
{ 5, 9, 1, 0, 6, "\0\0\1\0\0\0\1\0\1\0\0\0\0\0\0\0\1\1\0\0\1\0\0\1\0\0\1\1\1\0\1\0\0\1\0\1\0\0\1\0\0\1\1\0\1" },
/* 227 */
{ 5, 9, 1, 0, 6, "\0\1\0\1\0\1\0\1\0\0\0\0\0\0\0\0\1\1\0\0\1\0\0\1\0\0\1\1\1\0\1\0\0\1\0\1\0\0\1\0\0\1\1\0\1" },
/* 228 */
{ 5, 8, 1, 0, 6, "\0\1\0\1\0\0\0\0\0\0\0\1\1\0\0\1\0\0\1\0\0\1\1\1\0\1\0\0\1\0\1\0\0\1\0\0\1\1\0\1" },
/* 229 */
{ 5, 9, 1, 0, 6, "\0\0\1\0\0\0\1\0\1\0\0\0\1\0\0\0\1\1\0\0\1\0\0\1\0\0\1\1\1\0\1\0\0\1\0\1\0\0\1\0\0\1\1\0\1" },
/* 230 */
{ 8, 6, 1, 0, 9, "\0\1\1\0\1\1\0\0\1\0\0\1\0\0\1\0\0\1\1\1\1\1\1\0\1\0\0\1\0\0\0\0\1\0\0\1\1\0\0\1\0\1\1\0\1\1\1\0" },
/* 231 */
{ 4, 9, 1, -3, 5, "\0\1\1\0\1\0\0\1\1\0\0\0\1\0\0\0\1\0\0\1\0\1\1\0\0\1\0\0\0\0\1\0\1\1\1\0" },
/* 232 */
{ 5, 9, 1, 0, 6, "\0\1\0\0\0\0\0\1\0\0\0\0\0\0\0\0\1\1\0\0\1\0\0\1\0\1\1\1\1\0\1\0\0\0\0\1\1\0\0\1\0\1\1\1\0" },
/* 233 */
{ 5, 9, 1, 0, 6, "\0\0\1\0\0\0\1\0\0\0\0\0\0\0\0\0\1\1\0\0\1\0\0\1\0\1\1\1\1\0\1\0\0\0\0\1\1\0\0\1\0\1\1\1\0" },
/* 234 */
{ 5, 9, 1, 0, 6, "\0\0\1\0\0\0\1\0\1\0\0\0\0\0\0\0\1\1\0\0\1\0\0\1\0\1\1\1\1\0\1\0\0\0\0\1\1\0\0\1\0\1\1\1\0" },
/* 235 */
{ 5, 8, 1, 0, 6, "\0\1\0\1\0\0\0\0\0\0\0\1\1\0\0\1\0\0\1\0\1\1\1\1\0\1\0\0\0\0\1\1\0\0\1\0\1\1\1\0" },
/* 236 */
{ 3, 9, 0, 0, 3, "\1\0\0\0\1\0\0\0\0\1\1\0\0\1\0\0\1\0\0\1\0\0\1\0\1\1\1" },
/* 237 */
{ 3, 9, 0, 0, 3, "\0\1\0\1\0\0\0\0\0\1\1\0\0\1\0\0\1\0\0\1\0\0\1\0\1\1\1" },
/* 238 */
{ 3, 9, 0, 0, 3, "\0\1\0\1\0\1\0\0\0\1\1\0\0\1\0\0\1\0\0\1\0\0\1\0\1\1\1" },
/* 239 */
{ 3, 8, 0, 0, 3, "\1\0\1\0\0\0\1\1\0\0\1\0\0\1\0\0\1\0\0\1\0\1\1\1" },
/* 240 */
{ 4, 9, 1, 0, 6, "\0\1\0\0\0\1\1\1\1\0\1\0\0\1\1\1\1\0\0\1\1\0\0\1\1\0\0\1\1\0\0\1\0\1\1\0" },
/* 241 */
{ 6, 9, 0, 0, 6, "\0\0\1\0\1\0\0\1\0\1\0\0\0\0\0\0\0\0\1\0\1\1\0\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\1\1\1\0\1\1" },
/* 242 */
{ 4, 9, 1, 0, 6, "\0\1\0\0\0\0\1\0\0\0\0\0\0\1\1\0\1\0\0\1\1\0\0\1\1\0\0\1\1\0\0\1\0\1\1\0" },
/* 243 */
{ 4, 9, 1, 0, 6, "\0\0\1\0\0\1\0\0\0\0\0\0\0\1\1\0\1\0\0\1\1\0\0\1\1\0\0\1\1\0\0\1\0\1\1\0" },
/* 244 */
{ 4, 9, 1, 0, 6, "\0\0\1\0\0\1\0\1\0\0\0\0\0\1\1\0\1\0\0\1\1\0\0\1\1\0\0\1\1\0\0\1\0\1\1\0" },
/* 245 */
{ 4, 9, 1, 0, 6, "\0\1\0\1\1\0\1\0\0\0\0\0\0\1\1\0\1\0\0\1\1\0\0\1\1\0\0\1\1\0\0\1\0\1\1\0" },
/* 246 */
{ 4, 8, 1, 0, 6, "\1\0\1\0\0\0\0\0\0\1\1\0\1\0\0\1\1\0\0\1\1\0\0\1\1\0\0\1\0\1\1\0" },
/* 247 */
{ 5, 5, 1, 1, 7, "\0\0\1\0\0\0\0\0\0\0\1\1\1\1\1\0\0\0\0\0\0\0\1\0\0" },
/* 248 */
{ 6, 7, 0, -1, 6, "\0\0\1\1\0\1\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\1\1\0\0\1\0\0\0\0\0" },
/* 249 */
{ 6, 9, 0, 0, 6, "\0\0\1\0\0\0\0\0\0\1\0\0\0\0\0\0\0\0\1\1\0\1\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\0\1\1\0\1" },
/* 250 */
{ 6, 9, 0, 0, 6, "\0\0\0\1\0\0\0\0\1\0\0\0\0\0\0\0\0\0\1\1\0\1\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\0\1\1\0\1" },
/* 251 */
{ 6, 9, 0, 0, 6, "\0\0\1\0\0\0\0\1\0\1\0\0\0\0\0\0\0\0\1\1\0\1\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\0\1\1\0\1" },
/* 252 */
{ 6, 8, 0, 0, 6, "\0\1\0\1\0\0\0\0\0\0\0\0\1\1\0\1\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\1\0\0\0\1\1\0\1" },
/* 253 */
{ 6, 12, 0, -3, 6, "\0\0\0\0\1\0\0\0\0\1\0\0\0\0\0\0\0\0\1\1\0\0\1\1\0\1\0\0\1\0\0\1\0\1\1\0\0\1\0\1\0\0\0\0\1\1\0\0\0\0\1\0\0\0\0\0\1\0\0\0\0\1\0\0\0\0\1\1\0\0\0\0" },
/* 254 */
{ 5, 12, 0, -3, 6, "\1\1\0\0\0\0\1\0\0\0\0\1\0\0\0\0\1\1\1\0\0\1\0\0\1\0\1\0\0\1\0\1\0\0\1\0\1\0\0\1\0\1\1\1\0\0\1\0\0\0\0\1\0\0\0\1\1\1\0\0" },
/* 255 */
{ 6, 11, 0, -3, 6, "\0\1\0\0\1\0\0\0\0\0\0\0\1\1\0\0\1\1\0\1\0\0\1\0\0\1\0\1\1\0\0\1\0\1\0\0\0\0\1\1\0\0\0\0\1\0\0\0\0\0\1\0\0\0\0\1\0\0\0\0\1\1\0\0\0\0" }

};

struct font pbm_defaultBdffont = { 14, 15, -1, -3, {
NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
glBdf+ 0, glBdf+ 1, glBdf+ 2, glBdf+ 3, glBdf+ 4, glBdf+ 5,
glBdf+ 6, glBdf+ 7, glBdf+ 8, glBdf+ 9, glBdf+10, glBdf+11,
glBdf+12, glBdf+13, glBdf+14, glBdf+15, glBdf+16, glBdf+17,
glBdf+18, glBdf+19, glBdf+20, glBdf+21, glBdf+22, glBdf+23,
glBdf+24, glBdf+25, glBdf+26, glBdf+27, glBdf+28, glBdf+29,
glBdf+30, glBdf+31, glBdf+32, glBdf+33, glBdf+34, glBdf+35,
glBdf+36, glBdf+37, glBdf+38, glBdf+39, glBdf+40, glBdf+41,
glBdf+42, glBdf+43, glBdf+44, glBdf+45, glBdf+46, glBdf+47,
glBdf+48, glBdf+49, glBdf+50, glBdf+51, glBdf+52, glBdf+53,
glBdf+54, glBdf+55, glBdf+56, glBdf+57, glBdf+58, glBdf+59,
glBdf+60, glBdf+61, glBdf+62, glBdf+63, glBdf+64, glBdf+65,
glBdf+66, glBdf+67, glBdf+68, glBdf+69, glBdf+70, glBdf+71,
glBdf+72, glBdf+73, glBdf+74, glBdf+75, glBdf+76, glBdf+77,
glBdf+78, glBdf+79, glBdf+80, glBdf+81, glBdf+82, glBdf+83,
glBdf+84, glBdf+85, glBdf+86, glBdf+87, glBdf+88, glBdf+89,
glBdf+90, glBdf+91, glBdf+92, glBdf+93, glBdf+94,
NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 
glBdf+95,  glBdf+96,  glBdf+97,  glBdf+98,  glBdf+99,  glBdf+100,
glBdf+101, glBdf+102, glBdf+103, glBdf+104, glBdf+105, glBdf+106,
glBdf+107, glBdf+108, glBdf+109, glBdf+110, glBdf+111, glBdf+112,
glBdf+113, glBdf+114, glBdf+115, glBdf+116, glBdf+117, glBdf+118,
glBdf+119, glBdf+120, glBdf+121, glBdf+122, glBdf+123, glBdf+124,
glBdf+125, glBdf+126, glBdf+127, glBdf+128, glBdf+129, glBdf+130,
glBdf+131, glBdf+132, glBdf+133, glBdf+134, glBdf+135, glBdf+136,
glBdf+137, glBdf+138, glBdf+139, glBdf+140, glBdf+141, glBdf+142,
glBdf+143, glBdf+144, glBdf+145, glBdf+146, glBdf+147, glBdf+148,
glBdf+149, glBdf+150, glBdf+151, glBdf+152, glBdf+153, glBdf+154,
glBdf+155, glBdf+156, glBdf+157, glBdf+158, glBdf+159, glBdf+160,
glBdf+161, glBdf+162, glBdf+163, glBdf+164, glBdf+165, glBdf+166,
glBdf+167, glBdf+168, glBdf+169, glBdf+170, glBdf+171, glBdf+172,
glBdf+173, glBdf+174, glBdf+175, glBdf+176, glBdf+177, glBdf+178,
glBdf+179, glBdf+180, glBdf+181, glBdf+182, glBdf+183, glBdf+184,
glBdf+185, glBdf+186, glBdf+187, glBdf+188, glBdf+189  }
};


