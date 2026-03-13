// port 0 to port 13
// port 2 to port 14
#include <reg51.h>

#define led P0;
sbit sw = P2^0;

void delay_ms(unsigned int i);

void main(void)
{
    unsigned char i;
    while(1)
    {
        if (sw == 0)
        {
            for (i = 0; i < 4; i++)
            {
                led |= (1 << i);
                delay_ms(200);
            }
        }
    }
}

void delay_ms(unsigned int i)
{
    unsigned int j;

    while (i-- > 0)
    {
        for (j = 0; j < 500; j++)
        {
            ;
        }
    }
}