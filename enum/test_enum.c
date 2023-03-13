#include <stdio.h>

int main(int argc, char *argv[]) {

    enum yytokentype
        {
            YYEMPTY = -2,
            YYEOF = 0,                     /* "end of file"  */
            YYerror = 256,                 /* error  */
            YYUNDEF = 257,                 /* "invalid token"  */
            EOL = 258,                     /* EOL  */
            T_number = 259,                /* T_number  */
            T_SI_prefix = 260,             /* T_SI_prefix  */
            T_length = 261,                /* T_length  */
            T_mass = 262,                  /* T_mass  */
            T_time = 263,                  /* T_time  */
            T_current = 264,               /* T_current  */
            T_temp = 265,                  /* T_temp  */
            T_lumi = 266,                  /* T_lumi  */
            T_mole = 267,                  /* T_mole  */
            T_freq = 268,                  /* T_freq  */
            T_ang_rad = 269,               /* T_ang_rad  */
            T_ang_deg = 270,               /* T_ang_deg  */
            T_solid_ang = 271,             /* T_solid_ang  */
            T_Jansky = 272,                /* T_Jansky  */
            NEG = 273,                     /* NEG  */
            POS = 274                      /* POS  */
        };
    /* typedef enum yytokentype yytoken_kind_t; */

    printf("EOL = %d\n", EOL);

    return 0;
}
