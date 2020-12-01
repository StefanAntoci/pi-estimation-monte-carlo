#include <iostream>
#include <time.h>
#include <random>
#include <math.h>

using namespace std;
int main()
{
	unsigned int n = 512 * 512 * 128;
	float pi;
	//  serial verion
	clock_t cpu_start = clock();
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(0, 1.0);
	int count = 0;
	int temp = 0;
	for (unsigned int i = 0; i < n; i++) {
		/*int temp = 0;
		while (temp < m) {
			float x = distribution(generator);
			float y = distribution(generator);
			float r = x * x + y * y; 

			if (r <= 1) {
				count++;
			}
			temp++;
		}*/
		float x = distribution(generator);
		float y = distribution(generator);
		float r = x * x + y * y;

		if (r <= 1) {
			count++;
		}
		temp++;
		if (i % 2000 == 0) {
			pi = 4.0 * count / (temp);
			cout << "Approximate pi calculated on CPU is: " << pi << endl;
		}
	}
	clock_t cpu_stop = clock();
	std::cout << "Approximate pi calculated on CPU is: " << pi << " and calculation took " << (double)(cpu_stop - cpu_start) / CLOCKS_PER_SEC << std::endl;



}