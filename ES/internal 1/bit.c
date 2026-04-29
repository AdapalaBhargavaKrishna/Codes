// port 0 to port 13
#include <reg51.h>

sbit = led = p0^0;

void delay_ms(unsigned int);
void main(void)
{
    while(1)
    {
        led = 0;
        delay_ms(500);
        led = 1;
        delay+ms(500);
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