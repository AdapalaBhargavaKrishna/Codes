#include <reg51.h>
#define sevensegment_data P1

sbit DISP1_se1 = P0^3;
sbit DISP2_se1 = P0^2;
sbit DISP3_se1 = P0^1;
sbit DISP4_se1 = P0^0;
sbit lcd_back_light = P0^7;

void delay_ms(unsigned int);

void main(void)
{
    while(1)
    {
        DISP1_se1 = 0;
        sevensegment_data = 0x76;
        delay_ms(2);
        DISP1_se1=1;

        DISP2_se1 = 0;
        sevensegment_data = 0x77;
        delay_ms(2);
        DISP2_se1=1;
        
        DISP3_se1 = 0;
        sevensegment_data = 0x5F;
        delay_ms(2);
        DISP3_se1=1;

        DISP4_se1 = 0;
        sevensegment_data = 0x82;
        delay_ms(2);
        DISP4_se1=1;
    }
}


void delay_ms(unsigned int itime)
{
unsigned int i, j;
for (i =0; i < itime; i++)
for (j =0; j < 100; j++)
{
    ;
}
}