// port 0 to port 13
#include <reg51.h>
#define led P0

void delay_ms(unsigned int);

void main(void)
{
    led = 0x60;
    delay_ms(500);
    led = 0x04;
    delay_ms(500);
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