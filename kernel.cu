// ConsoleApplication1.cpp : Defines the entry point for the console application.
// нужно сделать будет когда-нибудь, если все будет работать:
//*** i + offset* j + offset2*k заменить на l, общий индекс
//*** в расчете квазискорости и Т, и, может быть, еще где объединить те аппроксимации на границах
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <omp.h>

#include <iostream> 

#ifdef _WIN32
#include "windows.h"
#endif

#include <stdio.h> 
#include <string> 
#include <fstream> 
#include <iomanip> 
#include <sstream> 
#include <cstring> 
#include <cmath>
#include <algorithm> 

#include <cuda.h>


#define Pi 3.1415926535897932384626433832795
#define pause system("pause");
#define timer timer2 = clock()/CLOCKS_PER_SEC; 	cout << "time (seconds)= " << (timer2 - timer1) << endl;
#define cudaCheckError() {                                          \
	cudaError_t e = cudaGetLastError();                                \
if (e != cudaSuccess) {\
	printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));           \
	exit(0); \
}                                                                 \
}


//CPU functions and variables 
using namespace std;


double maxval(double* f, unsigned int n)
{
	double max = abs(f[0]);

	for (unsigned int i = 0; i < n; i++) {
		if (abs(f[i])>max)
		{
			max = abs(f[i]);
		}
	}
	return max;
}
double minval(double* f, unsigned int n)
{
	double min = abs(f[0]);

	for (unsigned int i = 0; i < n; i++) {
		if (abs(f[i])<min)
		{
			min = abs(f[i]);
		}
	}
	return min;
}
void maxval_index(double* f, unsigned int nx, unsigned int ny, unsigned int nz, unsigned int& mx, unsigned int& my, unsigned int& mz, double& max)
{
	max = f[0];
	int offset = nx + 1; int offset2 = (nx + 1) * (ny + 1);
	mx = 0; my = 0; mz = 0;


	for (unsigned int i = 0; i <= nx; i++) {
		for (unsigned int j = 0; j <= ny; j++) {
			for (unsigned int k = 0; k <= nz; k++) {
				if (abs(f[i + offset*j + offset2*k]) > abs(max))
				{
					max = (f[i + offset*j + offset2*k]);
					mx = i;
					my = j;
					mz = k;
				}
			}
		}
	}

}

void velocity(int nx, int ny, int nz, double Lx, double Lz, double vx[], double vy[], double vz[], double &Ek, double &Vmax) {
	double hx, hy, hz;
	int off = nx + 1;
	int off2 = (nx + 1) * (ny + 1);
	double V = 0;
	Ek = 0.0; Vmax = 0.0;


	hy = 1.0 / ny;
	hx = hy; hz = hy; //!
	//cout << hx << " " << hy <<" " << hz << " " << endl;
	//cout << nx << " " << ny << " " << nz << " " << endl;

	for (int qx = 1; qx <= nx - 1; qx++) {
		for (int qy = 1; qy <= ny - 1; qy++) {
			for (int qz = 1; qz <= nz - 1; qz++) {
				V = +vx[qx + off * qy + off2*qz] * vx[qx + off * qy + off2*qz]
					+ vy[qx + off * qy + off2*qz] * vy[qx + off * qy + off2*qz]
					+ vz[qx + off * qy + off2*qz] * vz[qx + off * qy + off2*qz];
				Ek += V;
				if (sqrt(V) > Vmax) Vmax = sqrt(V);
			}
		}
	}

	Ek = Ek / 2.0 * hx * hy * hz;


}



void write_fields(int nx, int ny, int nz, double hx, double hy, double hz, int off, int off2, int size_b, int iter, double tau, double Ra,
	double *vx_h, double *vy_h, double *vz_h, double *T_h, double *C1_h, double *C2_h, double *p_h,
	double *vx_d, double *vy_d, double *vz_d, double *T_d, double *C1_d, double *C2_d, double *p_d) {

	//cudaMemcpy(T_h, T_d, size_b, cudaMemcpyDeviceToHost);
	//cudaMemcpy(C1_h, C1_d, size_b, cudaMemcpyDeviceToHost);
	//cudaMemcpy(C2_h, C2_d, size_b, cudaMemcpyDeviceToHost);
	//cudaMemcpy(p_h, p_d, size_b, cudaMemcpyDeviceToHost);
#ifdef __linux__
	string folder = "fields/";
#endif

#ifdef _WIN32
	string folder = "fields\\";
#endif






	stringstream ss, ss2; string str;
	ss.str(""); ss.clear();
	//ss << setprecision(15);
	ss << iter*tau;
	ss2 << Ra;
	str = "fields/" + ss.str() + " " + ss2.str() + ".txt";



	ofstream all((folder + ss.str() + " " + ss2.str() + " all" + ".txt").c_str());
	ofstream xy2((folder + ss.str() + " " + ss2.str() + " xy2_mid.txt").c_str());
	ofstream xy4((folder + ss.str() + " " + ss2.str() + " xy2_quoter.txt").c_str());
	ofstream xy34((folder + ss.str() + " " + ss2.str() + " xy2_3quoters.txt").c_str());


	ofstream yz2((folder + ss.str() + " " + ss2.str() + " yz2_mid.txt").c_str());
	ofstream yz4((folder + ss.str() + " " + ss2.str() + " yz2_quoter.txt").c_str());
	ofstream yz34((folder + ss.str() + " " + ss2.str() + " yz2_3quoters.txt").c_str());


	ofstream xz2((folder + ss.str() + " " + ss2.str() + " xz2_mid.txt").c_str());
	ofstream xz4((folder + ss.str() + " " + ss2.str() + " xz2_quoter.txt").c_str());
	ofstream xz34((folder + ss.str() + " " + ss2.str() + " xz2_3quoters.txt").c_str());




	xy2 << "#x, y, T, vx, vy, vz" << endl;
	xy4 << "#x, y, T, vx, vy, vz" << endl;
	xy34 << "#x, y, T, vx, vy, vz" << endl;
	for (int i = 0; i <= nx; i++) {
		for (int j = 0; j <= ny; j++) {
			xy2 << i*hx << " " << j*hy << " " << T_h[i + off*j + off2*(nz) / 2] << " " << vx_h[i + off*j + off2*(nz) / 2] << " " << vy_h[i + off*j + off2*(nz) / 2] << " " << vz_h[i + off*j + off2*(nz) / 2] << " " << C2_h[i + off*j + off2*(nz) / 2] << endl;
			xy4 << i*hx << " " << j*hy << " " << T_h[i + off*j + off2*(nz) / 4] << " " << vx_h[i + off*j + off2*(nz) / 4] << " " << vy_h[i + off*j + off2*(nz) / 4] << " " << vz_h[i + off*j + off2*(nz) / 4] << " " << C2_h[i + off*j + off2*(nz) / 4] << endl;
			xy34 << i*hx << " " << j*hy << " " << T_h[i + off*j + off2*(nz) / 4 * 3] << " " << vx_h[i + off*j + off2*(nz) / 4 * 3] << " " << vy_h[i + off*j + off2*(nz) / 4 * 3] << " " << vz_h[i + off*j + off2*(nz) / 4 * 3] << " " << C2_h[i + off*j + off2*(nz) / 4 * 3] << endl;
		}
	}

	yz2 << "#z, y, T, vz, vy, vx" << endl;
	yz4 << "#z, y, T, vz, vy, vx" << endl;
	yz34 << "#z, y, T, vz, vy, vx" << endl;
	for (int k = 0; k <= nz; k++) {
		for (int j = 0; j <= ny; j++) {
			yz2 << k*hz << " " << j*hy << " " << T_h[(nx) / 2 + off*j + off2*k] << " " << vz_h[(nx) / 2 + off*j + off2*k] << " " << vy_h[(nx) / 2 + off*j + off2*k] << " " << vx_h[(nx) / 2 + off*j + off2*k] << " " << C2_h[(nx) / 2 + off*j + off2*k] << endl;
			yz4 << k*hz << " " << j*hy << " " << T_h[(nx) / 4 + off*j + off2*k] << " " << vz_h[(nx) / 4 + off*j + off2*k] << " " << vy_h[(nx) / 4 + off*j + off2*k] << " " << vx_h[(nx) / 4 + off*j + off2*k] << " " << C2_h[(nx) / 4 + off*j + off2*k] << endl;
			yz34 << k*hz << " " << j*hy << " " << T_h[(nx) / 4 * 3 + off*j + off2*k] << " " << vz_h[(nx) / 4 * 3 + off*j + off2*k] << " " << vy_h[(nx) / 4 * 3 + off*j + off2*k] << " " << vx_h[(nx) / 4 * 3 + off*j + off2*k] << " " << C2_h[(nx) / 4 * 3 + off*j + off2*k] << endl;
		}
	}

	xz2 << "#x, z, T, vx, vz, vy" << endl;
	xz4 << "#x, z, T, vx, vz, vy" << endl;
	xz34 << "#x, z, T, vx, vz, vy" << endl;

	for (int i = 0; i <= nx; i++) {
		for (int k = 0; k <= nz; k++) {
			xz2 << i*hx << " " << k*hz << " " << T_h[i + off*(ny) / 2 + off2*k] << " " << vx_h[i + off*(ny) / 2 + off2*k] << " " << vz_h[i + off*(ny) / 2 + off2*k] << " " << vy_h[i + off*(ny) / 2 + off2*k] << " " << C2_h[i + off*(ny) / 2 + off2*k] << endl;
			xz4 << i*hx << " " << k*hz << " " << T_h[i + off*(ny) / 4 + off2*k] << " " << vx_h[i + off*(ny) / 4 + off2*k] << " " << vz_h[i + off*(ny) / 4 + off2*k] << " " << vy_h[i + off*(ny) / 4 + off2*k] << endl;
			xz34 << i*hx << " " << k*hz << " " << T_h[i + off*(ny) / 4 * 3 + off2*k] << " " << vx_h[i + off*(ny) / 4 * 3 + off2*k] << " " << vz_h[i + off*(ny) / 4 * 3 + off2*k] << " " << vy_h[i + off*(ny) / 4 * 3 + off2*k] << endl;
		}
	}

	all << "#x, y, z, vx, vy, vz, C1, C2, T, p" << endl;
	all << "#iter= " << iter << " Ra= " << Ra << endl;
	//all << setprecision(16) << fixed;
	for (int i = 0; i <= nx; i++) {
		for (int j = 0; j <= ny; j++) {
			for (int k = 0; k <= nz; k++) {
				all << i*hx << " " << j*hy << " " << k*hz << " " << vx_h[i + off*j + off2*k] << " " << vy_h[i + off*j + off2*k] << " " << vz_h[i + off*j + off2*k] << " " << C1_h[i + off*j + off2*k] << " " << C2_h[i + off*j + off2*k] << " " << T_h[i + off*j + off2*k] << " " << p_h[i + off*j + off2*k] << endl;
			}
		}
	}
}

void backup(int nx, int ny, int nz, double hx, double hy, double hz, int off, int off2, int size_b, int iter, double tau, double Ra,
	double *vx_h, double *vy_h, double *vz_h, double *T_h, double *C1_h, double *C2_h, double *p_h,
	double *vx_d, double *vy_d, double *vz_d, double *T_d, double *C1_d, double *C2_d, double *p_d) {


	ofstream all("all.txt");


	all << "#x, y, z, vx, vy, vz, C1, C2, T, p" << endl;
	all << "#iter= " << iter << " Ra= " << Ra << endl;
	//all << setprecision(16) << fixed;
	for (int i = 0; i <= nx; i++) {
		for (int j = 0; j <= ny; j++) {
			for (int k = 0; k <= nz; k++) {
				all << i*hx << " " << j*hy << " " << k*hz << " " << vx_h[i + off*j + off2*k] << " " << vy_h[i + off*j + off2*k] << " " << vz_h[i + off*j + off2*k] << " " << C1_h[i + off*j + off2*k] << " " << C2_h[i + off*j + off2*k] << " " << T_h[i + off*j + off2*k] << " " << p_h[i + off*j + off2*k] << endl;
			}
		}
	}
}


double max_vertical_difference(double* f, unsigned int nx, unsigned int ny, unsigned nz) {
	double max = 0, dif = 0;
	unsigned int offset = nx + 1;
	unsigned int offset2 = (ny + 1)*(nx + 1);

	for (int i = 0; i <= nx; i++){
		for (int k = 0; k <= nz; k++){
			dif = f[i + offset2*k] - f[i + offset*ny + offset2*k];
			dif = abs(dif);
			if (dif > max) max = dif;
		}
	}

	return max;
}

double Nu_y_down(double *f, unsigned int nx, unsigned int ny, unsigned int nz, double hx, double hy, double hz) {
	double N = 0;
	unsigned int offset = nx + 1;
	unsigned int offset2 = (ny + 1)*(nx + 1);
	double S = nx * hx * nz * hz;

	N += abs(f[0 + offset + offset2 * 0] - f[0 + offset2 * 0]) / hy;
	N += abs(f[0 + offset + offset2 * nz] - f[0 + offset2 * nz]) / hy;
	N += abs(f[nx + offset + offset2 * 0] - f[nx + offset2 * 0]) / hy;
	N += abs(f[nx + offset + offset2 * nz] - f[nx + offset2 * nz]) / hy;

	for (unsigned int i = 1; i <= nx - 1; i++)
		N += 2 * abs(f[i + offset + offset2 * 0] - f[i + offset2 * 0]) / hy;

	for (unsigned int i = 1; i <= nx - 1; i++)
		N += 2 * abs(f[i + offset + offset2 * nz] - f[i + offset2 * nz]) / hy;

	for (unsigned int k = 1; k <= nz - 1; k++)
		N += 2 * abs(f[0 + offset + offset2 * k] - f[0 + offset2 * k]) / hy;
	for (unsigned int k = 1; k <= nz - 1; k++)
		N += 2 * abs(f[nx + offset + offset2 * k] - f[nx + offset2 * k]) / hy;


	for (unsigned int i = 1; i <= nx - 1; i++) {
		for (unsigned int k = 1; k <= nz - 1; k++) {
			N += 4 * abs(f[i + offset + offset2 * k] - f[i + offset2 * k]) / hy;
		}
	}

	N = N  * (hx)* (hz) / 4 / S;
	return N;
}
double Nu_y_top(double *f, unsigned int nx, unsigned int ny, unsigned int nz, double hx, double hy, double hz) {
	double N = 0;
	unsigned int offset = nx + 1;
	unsigned int offset2 = (ny + 1)*(nx + 1);
	double S = nx * hx * nz * hz;


	N += abs(f[0 + offset*ny + offset2 * 0] - f[0 + offset*(ny - 1) + offset2 * 0]) / hy;
	N += abs(f[0 + offset*ny + offset2 * nz] - f[0 + offset*(ny - 1) + offset2 * nz]) / hy;
	N += abs(f[nx + offset*ny + offset2 * 0] - f[nx + offset*(ny - 1) + offset2 * 0]) / hy;
	N += abs(f[nx + offset*ny + offset2 * nz] - f[nx + offset*(ny - 1) + offset2 * nz]) / hy;

	for (unsigned int i = 1; i <= nx - 1; i++)
		N += 2 * abs(f[i + offset*ny + offset2 * 0] - f[i + offset*(ny - 1) + offset2 * 0]) / hy;

	for (unsigned int i = 1; i <= nx - 1; i++)
		N += 2 * abs(f[i + offset*ny + offset2 * nz] - f[i + offset*(ny - 1) + offset2 * nz]) / hy;

	for (unsigned int k = 1; k <= nz - 1; k++)
		N += 2 * abs(f[0 + offset*ny + offset2 * k] - f[0 + offset*(ny - 1) + offset2 * k]) / hy;
	for (unsigned int k = 1; k <= nz - 1; k++)
		N += 2 * abs(f[nx + offset*ny + offset2 * k] - f[nx + offset*(ny - 1) + offset2 * k]) / hy;


	for (unsigned int i = 1; i <= nx - 1; i++) {
		for (unsigned int k = 1; k <= nz - 1; k++) {
			N += 4 * abs(f[i + offset*ny + offset2 * k] - f[i + offset*(ny - 1) + offset2 * k]) / hy;
		}
	}

	N = N  * (hx)* (hz) / 4 / S;
	return N;
}

double Nu_x_left(double *f, unsigned int nx, unsigned int ny, unsigned int nz, double hx, double hy, double hz) {
	double N = 0;
	unsigned int offset = nx + 1;
	unsigned int offset2 = (ny + 1)*(nx + 1);
	double S = ny * hy * nz * hz;


	N += abs(f[1 + offset * 0 + offset2 * 0] - f[0 + offset * 0 + offset2 * 0]) / hx;
	N += abs(f[1 + offset*ny + offset2 * 0] - f[0 + offset*(ny)+offset2 * 0]) / hx;
	N += abs(f[1 + offset * 0 + offset2 * nz] - f[0 + offset*(0) + offset2 * nz]) / hx;
	N += abs(f[1 + offset*ny + offset2 * nz] - f[0 + offset*(ny)+offset2 * nz]) / hx;

	for (unsigned int j = 1; j <= ny - 1; j++)
		N += 2 * abs(f[1 + offset*j + offset2 * 0] - f[0 + offset*j + offset2 * 0]) / hx;

	for (unsigned int j = 1; j <= ny - 1; j++)
		N += 2 * abs(f[1 + offset*j + offset2 * nz] - f[0 + offset*j + offset2 * nz]) / hx;


	for (unsigned int k = 1; k <= nz - 1; k++)
		N += 2 * abs(f[1 + offset * 0 + offset2 * k] - f[0 + offset * 0 + offset2 * k]) / hx;
	for (unsigned int k = 1; k <= nz - 1; k++)
		N += 2 * abs(f[1 + offset*ny + offset2 * k] - f[0 + offset*ny + offset2 * k]) / hx;


	for (unsigned int j = 1; j <= ny - 1; j++) {
		for (unsigned int k = 1; k <= nz - 1; k++) {
			N += 4 * abs(f[1 + offset*j + offset2 * k] - f[0 + offset*j + offset2 * k]) / hx;
		}
	}

	N = N  * (hz)* (hy) / 4 / S;
	return N;
}
double Nu_x_right(double *f, unsigned int nx, unsigned int ny, unsigned int nz, double hx, double hy, double hz) {
	double N = 0;
	unsigned int offset = nx + 1;
	unsigned int offset2 = (ny + 1)*(nx + 1);
	double S = ny * hy * nz * hz;


	N += abs(f[nx + offset * 0 + offset2 * 0] - f[nx - 1 + offset * 0 + offset2 * 0]) / hy;
	N += abs(f[nx + offset*ny + offset2 * 0] - f[nx - 1 + offset*(ny)+offset2 * 0]) / hy;
	N += abs(f[nx + offset * 0 + offset2 * nz] - f[nx - 1 + offset*(0) + offset2 * nz]) / hy;
	N += abs(f[nx + offset*ny + offset2 * nz] - f[nx - 1 + offset*(ny)+offset2 * nz]) / hy;

	for (unsigned int j = 1; j <= ny - 1; j++)
		N += 2 * abs(f[nx + offset*j + offset2 * 0] - f[nx - 1 + offset*j + offset2 * 0]) / hy;
	for (unsigned int j = 1; j <= ny - 1; j++)
		N += 2 * abs(f[nx + offset*j + offset2 * nz] - f[nx - 1 + offset*j + offset2 * nz]) / hy;

	for (unsigned int k = 1; k <= nz - 1; k++)
		N += 2 * abs(f[nx + offset * 0 + offset2 * k] - f[nx - 1 + offset * 0 + offset2 * k]) / hy;
	for (unsigned int k = 1; k <= nz - 1; k++)
		N += 2 * abs(f[nx + offset*ny + offset2 * k] - f[nx - 1 + offset*ny + offset2 * k]) / hy;


	for (unsigned int j = 1; j <= ny - 1; j++) {
		for (unsigned int k = 1; k <= nz - 1; k++) {
			N += 4 * abs(f[nx + offset*j + offset2 * k] - f[nx - 1 + offset*j + offset2 * k]) / hy;
		}
	}

	N = N  * (hx)* (hz) / 4 / S;
	return N;
}


void angular_momentum(double *vx, double *vy, double *vz, unsigned int nx, unsigned int ny, unsigned int nz, double hx, double hy, double hz,
	double &AMx, double &AMy, double &AMz, double &AMabs)
{
	AMx = 0; AMy = 0; AMz = 0; AMabs = 0;
	double x0 = nx / 2 * hx;
	double y0 = ny / 2 * hy;
	double z0 = nz / 2 * hz;
	double x, y, z;
	unsigned int l;
	unsigned int offset = nx + 1;
	unsigned int offset2 = (ny + 1)*(nx + 1);

	for (int i = 1; i <= nx - 1; i++) {
		for (int j = 1; j <= ny - 1; j++) {
			for (int k = 1; k <= nz - 1; k++) {
				l = i + offset*j + offset2*k;
				x = i*hx; y = j*hy; z = k*hz;
				AMx += (z - z0) * vy[l] - (y - y0) * vz[l];
				AMy += (x - x0) * vz[l] - (z - z0) * vx[l];
				AMz += (y - y0) * vx[l] - (x - x0) * vy[l];
			}
		}
	}

	AMx *= hx*hy*hz;
	AMy *= hx*hy*hz;
	AMz *= hx*hy*hz;

	AMabs = sqrt(AMx*AMx + AMy*AMy + AMz*AMz);
}


//GPU functions and variables
__constant__ double hx_d, hy_d, hz_d, tau_d, Lx_d, Ly_d, Lz_d, Ra_d, Pr_d, tau_p_d;
__constant__ double Sc11_d, Sc12_d, Sc21_d, Sc22_d, psi1_d, psi2_d, psiS_d, Q_d;
__constant__ double sinA, cosA;
__constant__ unsigned int nx_d, ny_d, nz_d, n_d, offset, offset2, border_type;
__constant__ double eps0_d = 1e-5;
__constant__ double pi_d = 3.1415926535897932384626433832795;
__device__ double eps_d = 1.0; // psiav_d, psiav0_d = 0.0;
__device__ double dp;

__global__ void hello() {

	printf("\n thread x:%i y:%i, information copied from device:\n", threadIdx.x, threadIdx.y);
	printf("Ra= %f Pr=%f \n", Ra_d, Pr_d);
	printf("psi1= %f psi2= %f psiS= %f\n", psi1_d, psi2_d, psiS_d);
	printf("Sc1= %f Sc2=%f \n", Sc11_d, Sc22_d);


	printf("hx= %f hy=%f hz=%f \n", hx_d, hy_d, hz_d);
	printf("tau= %20.16f  \n", tau_d);
	printf("nx= %i ny=%i nz=%i N=%i \n", nx_d, ny_d, nz_d, n_d);
	printf("Lx= %f Ly=%f Lz=%f \n", Lx_d, Ly_d, Lz_d);
	printf("offset= %i offset2=%i \n", offset, offset2);
	printf("sinA= %f cosA=%f \n", sinA, cosA);
	printf("border= %i \n", border_type);

	printf("\n");
}


__global__ void init(double *ux_d, double *uy_d, double *uz_d, double *vx_d, double *vy_d, double *vz_d,
	double *T_d, double *T0_d, double *C1_d, double *C10_d, double *C2_d, double *C20_d, double *p_d, double *p0_d)
{

	for (int i = 0; i <= nx_d; i++) {
		for (int j = 0; j <= ny_d; j++) {
			for (int k = 0; k <= nz_d; k++) {
				//T_h[i + off*j + off2*k] = 0.0;
				//T_h[i + off*j + off2*k] = 1.0 - hy*j;
				//if (i == 0 || i == nx || j == 0 || j == ny || k == 0 || k == nz)
				T_d[i + offset*j + offset2*k] = 1.0 - hy_d*j;
				C1_d[i + offset*j + offset2*k] = (1.0 - hy_d*j)*psi1_d;
				C2_d[i + offset*j + offset2*k] = (1.0 - hy_d*j)*psi2_d;
				T0_d[i + offset*j + offset2*k] = T_d[i + offset*j + offset2*k];
				C10_d[i + offset*j + offset2*k] = C1_d[i + offset*j + offset2*k];
				C20_d[i + offset*j + offset2*k] = C2_d[i + offset*j + offset2*k];
				vx_d[i + offset*j + offset2*k] = 0.0;
				vy_d[i + offset*j + offset2*k] = 0.0;
				vz_d[i + offset*j + offset2*k] = 0.0;
				ux_d[i + offset*j + offset2*k] = 0.0;
				uy_d[i + offset*j + offset2*k] = 0.0;
				uz_d[i + offset*j + offset2*k] = 0.0;
				p_d[i + offset*j + offset2*k] = 0.0;
				p0_d[i + offset*j + offset2*k] = 0.0;
			}
		}
	}

	T_d[10 + offset * 0 + offset2 * 10] = 2.0;


}

__global__ void quasi_velocity(double *ux_d, double *uy_d, double *uz_d, double *vx_d, double *vy_d, double *vz_d, double *T_d, double *T0_d, double *C1_d, double *C10_d, double *C2_d, double *C20_d) {

	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y*blockDim.y;
	unsigned int k = threadIdx.z + blockIdx.z*blockDim.z;
	unsigned int l = i + offset*j + offset2*k;
	if (i <= nx_d && j <= ny_d && k <= nz_d && l < n_d)
	{

		/*
		INNER
		*/
		if (i > 0 && i < nx_d && j > 0 && j < ny_d && k > 0 && k < nz_d)
		{

			//ux_d
			ux_d[l] = vx_d[l]
				+ tau_d * (
				-vx_d[l] * (vx_d[l + 1] - vx_d[l - 1]) / 2.0 / hx_d					//  (ux_d*d/dx) ux_d 
				- vy_d[l] * (vx_d[l + offset] - vx_d[l - offset]) / 2.0 / hy_d				//  (uy_d*d/dy) ux_d 
				- vz_d[l] * (vx_d[l + offset2] - vx_d[l - offset2]) / 2.0 / hz_d				//  (uy_d*d/dy) ux_d 

				+ (vx_d[l + 1] - 2.0*vx_d[l] + vx_d[l - 1]) / hx_d / hx_d				// (d2/dx2) ux_d
				+ (vx_d[l + offset] - 2.0*vx_d[l] + vx_d[l - offset]) / hy_d / hy_d			// (d2/dy2) ux_d   
				+ (vx_d[l + offset2] - 2.0*vx_d[l] + vx_d[l - offset2]) / hz_d / hz_d			// (d2/dz2) ux_d 

				+ Ra_d / Pr_d*(T0_d[l] + C10_d[l] + C20_d[l])*sinA
				);

			//uy_d
			uy_d[l] = vy_d[l]
				+ tau_d * (
				-vx_d[l] * (vy_d[l + 1] - vy_d[l - 1]) / 2.0 / hx_d					//  (ux_d*d/dx) uy_d
				- vy_d[l] * (vy_d[l + offset] - vy_d[l - offset]) / 2.0 / hy_d               //  (uy_d*d/dy) uy_d  
				- vz_d[l] * (vy_d[l + offset2] - vy_d[l - offset2]) / 2.0 / hz_d				//  (uy_d*d/dy) uy_d 

				+ (vy_d[l + 1] - 2.0*vy_d[l] + vy_d[l - 1]) / hx_d / hx_d				// (d2/dx2) uy_d
				+ (vy_d[l + offset] - 2.0*vy_d[l] + vy_d[l - offset]) / hy_d / hy_d			// (d2/dy2) uy_d   
				+ (vy_d[l + offset2] - 2.0*vy_d[l] + vy_d[l - offset2]) / hz_d / hz_d			// (d2/dz2) uy_d 

				+ Ra_d / Pr_d*(T0_d[l] + C10_d[l] + C20_d[l]) *cosA 						 // Ra/Pe*T_d
				);

			//uz_d
			uz_d[l] = vz_d[l]
				+ tau_d * (
				-vx_d[l] * (vz_d[l + 1] - vz_d[l - 1]) / 2.0 / hx_d					//  (ux_d*d/dx) uz_d 
				- vy_d[l] * (vz_d[l + offset] - vz_d[l - offset]) / 2.0 / hy_d				//  (uy_d*d/dy) uz_d 
				- vz_d[l] * (vz_d[l + offset2] - vz_d[l - offset2]) / 2.0 / hz_d				//  (uy_d*d/dy) uz_d 

				+ (vz_d[l + 1] - 2.0*vz_d[l] + vz_d[l - 1]) / hx_d / hx_d				// (d2/dx2) uz_d
				+ (vz_d[l + offset] - 2.0*vz_d[l] + vz_d[l - offset]) / hy_d / hy_d			// (d2/dy2) uz_d   
				+ (vz_d[l + offset2] - 2.0*vz_d[l] + vz_d[l - offset2]) / hz_d / hz_d			// (d2/dz2) uz_d 

				);

		}

		/*
		UP-DOWN
		*/

		// y
		else if (j == 0 && (i > 0 && i < nx_d && k > 0 && k < nz_d))
		{
			uy_d[l] = tau_d / hy_d / hy_d*(2.0 * vy_d[l] - 5.0 * vy_d[l + offset] + 4.0 * vy_d[l + offset * 2] - vy_d[l + offset * 3]) + tau_d* Ra_d / Pr_d*(T0_d[l] + C10_d[l] + C20_d[l]) * cosA;
			//ux_d[l] = tau_d / hy_d / hy_d*(2 * vx_d[l] - 5 * vx_d[l + offset] + 4 * vx_d[l + offset * 2] - vx_d[l + offset * 3]);
			//uz_d[l] = tau_d / hz_d / hz_d*(2 * vz_d[l] - 5 * vz_d[l + offset] + 4 * vz_d[l + offset * 2] - vz_d[l + offset * 3]);
		}
		else if (j == ny_d && (i > 0 && i < nx_d && k > 0 && k < nz_d))
		{
			uy_d[l] = tau_d / hy_d / hy_d*(2.0 * vy_d[l] - 5.0 * vy_d[l - offset] + 4.0 * vy_d[l - offset * 2] - vy_d[l - offset * 3]) + tau_d*Ra_d / Pr_d*(T0_d[l] + C10_d[l] + C20_d[l]) * cosA;
			//ux_d[l] = tau_d / hy_d / hy_d*(2 * vx_d[l] - 5 * vx_d[l - offset] + 4 * vx_d[l - offset * 2] - vx_d[l - offset * 3]);
			//uz_d[l] = tau_d / hy_d / hy_d*(2 * vz_d[l] - 5 * vz_d[l - offset] + 4 * vz_d[l - offset * 2] - vz_d[l - offset * 3]);
		}

		/*
		CLOSED
		*/

		// x
		else if (border_type == 0 && i == 0 && (j > 0 && j < ny_d && k > 0 && k < nz_d))
		{
			ux_d[l] = tau_d / hx_d / hx_d * (2.0 * vx_d[l] - 5.0 * vx_d[l + 1] + 4.0 * vx_d[l + 2] - vx_d[l + 3]) + tau_d*Ra_d / Pr_d*(T0_d[l] + C10_d[l] + C20_d[l]) * sinA;
			//uy_d[l] = tau_d / hx_d / hx_d * (2 * vy_d[l] - 5 * vy_d[l + 1] + 4 * vy_d[l + 2] - vy_d[l + 3]);
			//uz_d[l] = tau_d / hx_d / hx_d * (2 * vz_d[l] - 5 * vz_d[l + 1] + 4 * vz_d[l + 2] - vz_d[l + 3]);
		}
		else if (border_type == 0 && i == nx_d && (j > 0 && j < ny_d && k > 0 && k < nz_d))
		{
			ux_d[l] = tau_d / hx_d / hx_d * (2.0 * vx_d[l] - 5.0 * vx_d[l - 1] + 4.0 * vx_d[l - 2] - vx_d[l - 3]) + tau_d*Ra_d / Pr_d*(T0_d[l] + C10_d[l] + C20_d[l]) * sinA;
			//uy_d[l] = tau_d / hx_d / hx_d * (2 * vy_d[l] - 5 * vy_d[l - 1] + 4 * vy_d[l - 2] - vy_d[l - 3]);
			//uz_d[l] = tau_d / hx_d / hx_d * (2 * vz_d[l] - 5 * vz_d[l - 1] + 4 * vz_d[l - 2] - vz_d[l - 3]);
		}

		// z
		else if (border_type == 0 && k == 0 && (i > 0 && i < nx_d && j > 0 && j < ny_d))
		{
			//ux_d[l] = tau_d / hz_d / hz_d * (2 * vx_d[l] - 5 * vx_d[l + offset2] + 4 * vx_d[l + offset2 * 2] - vx_d[l + offset2 * 3]);
			//uy_d[l] = tau_d / hz_d / hz_d * (2 * vy_d[l] - 5 * vy_d[l + offset2] + 4 * vy_d[l + offset2 * 2] - vy_d[l + offset2 * 3]);
			uz_d[l] = tau_d / hz_d / hz_d * (2.0 * vz_d[l] - 5.0 * vz_d[l + offset2] + 4.0 * vz_d[l + offset2 * 2] - vz_d[l + offset2 * 3]);
		}
		else if (border_type == 0 && k == nz_d && (i > 0 && i < nx_d && j > 0 && j < ny_d))
		{
			//ux_d[l] = tau_d / hz_d / hz_d * (2 * vx_d[l] - 5 * vx_d[l - offset2] + 4 * vx_d[l - offset2 * 2] - vx_d[l - offset2 * 3]);
			//uy_d[l] = tau_d / hz_d / hz_d * (2 * vy_d[l] - 5 * vy_d[l - offset2] + 4 * vy_d[l - offset2 * 2] - vy_d[l - offset2 * 3]);
			uz_d[l] = tau_d / hz_d / hz_d * (2.0 * vz_d[l] - 5.0 * vz_d[l - offset2] + 4.0 * vz_d[l - offset2 * 2] - vz_d[l - offset2 * 3]);
		}


		/*
		PERIODIC
		*/

		// x

		else if (border_type == 1 && i == 0 && (j > 0 && j < ny_d && k > 0 && k < nz_d))
		{
			int ll = nx_d - 1 + offset*j + offset2*k;
			/*
			there is a slight possibility to come across the race condition effect while realizing parallel boundary condotions
			we think it is acceptabele to re-calculate some date once more to eliminate it
			ll index means the index on the opposite side but minus 1 (nx-1 here) as it goes to build a standard approach for parallel boundary conditions
			after we re-calculate the data for this index, nx-1, (that, in fact, was found above in the section for the whole inner nodes)
			*/

			ux_d[l] = vx_d[ll]
				+ tau_d * (
				-vx_d[ll] * (vx_d[ll + 1] - vx_d[ll - 1]) / 2.0 / hx_d					//  (ux_d*d/dx) ux_d
				- vy_d[ll] * (vx_d[ll + offset] - vx_d[ll - offset]) / 2.0 / hy_d				//  (uy_d*d/dy) ux_d
				- vz_d[ll] * (vx_d[ll + offset2] - vx_d[ll - offset2]) / 2.0 / hz_d				//  (uy_d*d/dy) ux_d

				+ (vx_d[ll + 1] - 2.0*vx_d[ll] + vx_d[ll - 1]) / hx_d / hx_d				// (d2/dx2) ux_d
				+ (vx_d[ll + offset] - 2.0*vx_d[ll] + vx_d[ll - offset]) / hy_d / hy_d			// (d2/dy2) ux_d
				+ (vx_d[ll + offset2] - 2.0*vx_d[ll] + vx_d[ll - offset2]) / hz_d / hz_d			// (d2/dz2) ux_d
				);

		}
		else if (border_type == 1 && i == nx_d && (j > 0 && j < ny_d && k > 0 && k < nz_d))
		{
			int ll = 1 + offset*j + offset2*k;
			ux_d[l] = vx_d[ll]
				+ tau_d * (
				-vx_d[ll] * (vx_d[ll + 1] - vx_d[ll - 1]) / 2.0 / hx_d					//  (ux_d*d/dx) ux_d
				- vy_d[ll] * (vx_d[ll + offset] - vx_d[ll - offset]) / 2.0 / hy_d				//  (uy_d*d/dy) ux_d
				- vz_d[ll] * (vx_d[ll + offset2] - vx_d[ll - offset2]) / 2.0 / hz_d				//  (uy_d*d/dy) ux_d

				+ (vx_d[ll + 1] - 2.0*vx_d[ll] + vx_d[ll - 1]) / hx_d / hx_d				// (d2/dx2) ux_d
				+ (vx_d[ll + offset] - 2.0*vx_d[ll] + vx_d[ll - offset]) / hy_d / hy_d			// (d2/dy2) ux_d
				+ (vx_d[ll + offset2] - 2.0*vx_d[ll] + vx_d[ll - offset2]) / hz_d / hz_d			// (d2/dz2) ux_d
				);
		}

		// z
		else if (border_type == 1 && k == 0 && (i > 0 && i < nx_d && j > 0 && j < ny_d))
		{
			int ll = i + offset*j + offset2*nz_d - offset2;

			uz_d[l] = vz_d[ll]
				+ tau_d * (
				-vx_d[ll] * (vz_d[ll + 1] - vz_d[ll - 1]) / 2.0 / hx_d					//  (ux_d*d/dx) uz_d
				- vy_d[ll] * (vz_d[ll + offset] - vz_d[ll - offset]) / 2.0 / hy_d				//  (uy_d*d/dy) uz_d
				- vz_d[ll] * (vz_d[ll + offset2] - vz_d[ll - offset2]) / 2.0 / hz_d				//  (uy_d*d/dy) uz_d

				+ (vz_d[ll + 1] - 2.0*vz_d[ll] + vz_d[ll - 1]) / hx_d / hx_d				// (d2/dx2) uz_d
				+ (vz_d[ll + offset] - 2.0*vz_d[ll] + vz_d[ll - offset]) / hy_d / hy_d			// (d2/dy2) uz_d
				+ (vz_d[ll + offset2] - 2.0*vz_d[ll] + vz_d[ll - offset2]) / hz_d / hz_d			// (d2/dz2) uz_d
				);

		}
		else if (border_type == 1 && k == nz_d && (i > 0 && i < nx_d && j > 0 && j < ny_d))
		{
			int ll = i + offset*j + offset2;

			uz_d[l] = vz_d[ll]
				+ tau_d * (
				-vx_d[ll] * (vz_d[ll + 1] - vz_d[ll - 1]) / 2.0 / hx_d					//  (ux_d*d/dx) uz_d
				- vy_d[ll] * (vz_d[ll + offset] - vz_d[ll - offset]) / 2.0 / hy_d				//  (uy_d*d/dy) uz_d
				- vz_d[ll] * (vz_d[ll + offset2] - vz_d[ll - offset2]) / 2.0 / hz_d				//  (uy_d*d/dy) uz_d

				+ (vz_d[ll + 1] - 2.0*vz_d[ll] + vz_d[ll - 1]) / hx_d / hx_d				// (d2/dx2) uz_d
				+ (vz_d[ll + offset] - 2.0*vz_d[ll] + vz_d[ll - offset]) / hy_d / hy_d			// (d2/dy2) uz_d
				+ (vz_d[ll + offset2] - 2.0*vz_d[ll] + vz_d[ll - offset2]) / hz_d / hz_d			// (d2/dz2) uz_d
				);

		}

		// corner points

		else if (i <= nx_d && j <= ny_d && k <= nz_d && l < n_d) {
			int ii = i + 1 - i / nx_d - ceil(i / (nx_d + 1.0));
			int jj = j + 1 - j / ny_d - ceil(j / (ny_d + 1.0));
			int kk = k + 1 - k / nz_d - ceil(k / (nz_d + 1.0));

			//ux_d[l] = 0;
			//uy_d[l] = 0;
			//uz_d[l] = 0;

			//ux_d[l] = ux_d[ii + offset*jj + offset2*kk];
			//uy_d[l] = uy_d[ii + offset*jj + offset2*kk];
			//uz_d[l] = uz_d[ii + offset*jj + offset2*kk];
		}


	}
}

__global__ void temperature(double *ux_d, double *uy_d, double *uz_d, double *vx_d, double *vy_d, double *vz_d, double *T_d, double *T0_d, double *C1_d, double *C10_d, double *C2_d, double *C20_d) {


	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y*blockDim.y;
	unsigned int k = threadIdx.z + blockIdx.z*blockDim.z;
	unsigned int l = i + offset*j + offset2*k;
	if (i <= nx_d && j <= ny_d && k <= nz_d && l < n_d)
	{

		/*
		INNER
		*/
		if (i > 0 && i < nx_d && j > 0 && j < ny_d && k > 0 && k < nz_d)
		{

			//temperature
			T_d[l] = T0_d[l]
				+ tau_d * (
				-vx_d[l] * (T0_d[l + 1] - T0_d[l - 1]) / (2.0 * hx_d)					// (ux_d*d/dx) T_d
				- vy_d[l] * (T0_d[l + offset] - T0_d[l - offset]) / (2.0 * hy_d)				// (uy_d*d/dy) T_d
				- vz_d[l] * (T0_d[l + offset2] - T0_d[l - offset2]) / (2.0 * hz_d)				// (uz_d*d/dy) T_d

				+ ((T0_d[l + 1] - 2.0*T0_d[l] + T0_d[l - 1]) / hx_d / hx_d 		// 1/Pr*(d2/dx2) T_d
				+ (T0_d[l + offset] - 2.0*T0_d[l] + T0_d[l - offset]) / hy_d / hy_d 	// 1/Pr*(d2/dy2) T_d
				+ (T0_d[l + offset2] - 2.0*T0_d[l] + T0_d[l - offset2]) / hz_d / hz_d) / Pr_d	// 1/Pr*(d2/dz2) T_d
				);

		}

		/*
		UP-DOWN
		*/

		// y
		else if (j == 0 && (i > 0 && i < nx_d && k > 0 && k < nz_d)) 		T_d[l] = (4.0*T0_d[l + offset] - T0_d[l + offset * 2]) / 3.0 - 2.0 / 3.0 *hy_d * Q_d; 
		else if (j == ny_d && (i > 0 && i < nx_d && k > 0 && k < nz_d))		T_d[l] = (4.0*T0_d[l - offset] - T0_d[l - offset * 2]) / 3.0 + 2.0 / 3.0 *hy_d * Q_d;

		/*
		CLOSED
		*/

		// x
		else if (border_type == 0 && i == 0 && (j > 0 && j < ny_d && k > 0 && k < nz_d)) 		T_d[l] = (4.0*T0_d[l + 1] - T0_d[l + 2]) / 3.0;
		else if (border_type == 0 && i == nx_d && (j > 0 && j < ny_d && k > 0 && k < nz_d))		T_d[l] = (4.0*T0_d[l - 1] - T0_d[l - 2]) / 3.0;

		// z
		else if (border_type == 0 && k == 0 && (i > 0 && i < nx_d && j > 0 && j < ny_d))		T_d[l] = (4.0*T0_d[l + offset2] - T0_d[l + offset2 * 2]) / 3.0;
		else if (border_type == 0 && k == nz_d && (i > 0 && i < nx_d && j > 0 && j < ny_d))		T_d[l] = (4.0*T0_d[l - offset2] - T0_d[l - offset2 * 2]) / 3.0;


		/*
		PERIODIC
		*/

		// x

		else if (border_type == 1 && i == 0 && (j > 0 && j < ny_d && k > 0 && k < nz_d))		T_d[l] = T0_d[nx_d - 1 + offset*j + offset2*k];
		else if (border_type == 1 && i == nx_d && (j > 0 && j < ny_d && k > 0 && k < nz_d))		T_d[l] = T0_d[1 + offset*j + offset2*k];

		// z
		else if (border_type == 1 && k == 0 && (i > 0 && i < nx_d && j > 0 && j < ny_d))		T_d[l] = T0_d[i + offset*j + offset2*nz_d - offset2];
		else if (border_type == 1 && k == nz_d && (i > 0 && i < nx_d && j > 0 && j < ny_d))		T_d[l] = T0_d[i + offset*j + offset2];


		// corner points

		else if (i <= nx_d && j <= ny_d && k <= nz_d && l < n_d) {
			int ii = i + 1 - i / nx_d - ceil(i / (nx_d + 1.0));
			int jj = j + 1 - j / ny_d - ceil(j / (ny_d + 1.0));
			int kk = k + 1 - k / nz_d - ceil(k / (nz_d + 1.0));

			T_d[l] = T0_d[ii + offset*jj + offset2*kk];
		}


	}
}

__global__ void concentration1(double *ux_d, double *uy_d, double *uz_d, double *vx_d, double *vy_d, double *vz_d, double *T_d, double *T0_d, double *C1_d, double *C10_d, double *C2_d, double *C20_d) {


	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y*blockDim.y;
	unsigned int k = threadIdx.z + blockIdx.z*blockDim.z;
	unsigned int l = i + offset*j + offset2*k;
	if (i <= nx_d && j <= ny_d && k <= nz_d && l < n_d)
	{

		/*
		INNER
		*/
		if (i > 0 && i < nx_d && j > 0 && j < ny_d && k > 0 && k < nz_d)
		{

			//C1
			C1_d[l] = C10_d[l]
				+ tau_d * (
				-vx_d[l] * (C10_d[l + 1] - C10_d[l - 1]) / (2.0 * hx_d)					// (ux_d*d/dx) C1_d
				- vy_d[l] * (C10_d[l + offset] - C10_d[l - offset]) / (2.0 * hy_d)				// (uy_d*d/dy) C1_d
				- vz_d[l] * (C10_d[l + offset2] - C10_d[l - offset2]) / (2.0 * hz_d)				// (uz_d*d/dy) C1_d

				+ ((C10_d[l + 1] - 2.0*C10_d[l] + C10_d[l - 1]) / hx_d / hx_d 	// 1/Pr*(d2/dx2) C1_d
				+ (C10_d[l + offset] - 2.0*C10_d[l] + C10_d[l - offset]) / hy_d / hy_d 	// 1/Pr*(d2/dy2) C1_d
				+ (C10_d[l + offset2] - 2.0*C10_d[l] + C10_d[l - offset2]) / hz_d / hz_d) / Sc11_d	// 1/Pr*(d2/dz2) C1_d

				- (+(T0_d[l + 1] - 2.0*T0_d[l] + T0_d[l - 1]) / hx_d / hx_d 		// 1/Pr*(d2/dx2) T_d
				+ (T0_d[l + offset] - 2.0*T0_d[l] + T0_d[l - offset]) / hy_d / hy_d 	// 1/Pr*(d2/dy2) T_d
				+ (T0_d[l + offset2] - 2.0*T0_d[l] + T0_d[l - offset2]) / hz_d / hz_d) / Sc11_d * psi1_d	// 1/Pr*(d2/dz2) T_d
				);

		}



		/*
		UP-DOWN
		*/

		// y
		else if (j == 0 && (i > 0 && i < nx_d && k > 0 && k < nz_d)) 		C1_d[l] = (4.0*C10_d[l + offset] - C10_d[l + offset * 2]) / 3.0 - 2.0 / 3.0 * hy_d*psi1_d*Q_d;
		else if (j == ny_d && (i > 0 && i < nx_d && k > 0 && k < nz_d))		C1_d[l] = (4.0*C10_d[l - offset] - C10_d[l - offset * 2]) / 3.0 + 2.0 / 3.0 * hy_d*psi1_d*Q_d;

		/*
		CLOSED
		*/

		// x
		else if (border_type == 0 && i == 0 && (j > 0 && j < ny_d && k > 0 && k < nz_d)) 		C1_d[l] = (4.0*C10_d[l + 1] - C10_d[l + 2]) / 3.0;
		else if (border_type == 0 && i == nx_d && (j > 0 && j < ny_d && k > 0 && k < nz_d))		C1_d[l] = (4.0*C10_d[l - 1] - C10_d[l - 2]) / 3.0;

		// z
		else if (border_type == 0 && k == 0 && (i > 0 && i < nx_d && j > 0 && j < ny_d))		C1_d[l] = (4.0*C10_d[l + offset2] - C10_d[l + offset2 * 2]) / 3.0;
		else if (border_type == 0 && k == nz_d && (i > 0 && i < nx_d && j > 0 && j < ny_d))		C1_d[l] = (4.0*C10_d[l - offset2] - C10_d[l - offset2 * 2]) / 3.0;


		/*
		PERIODIC
		*/

		// x

		else if (border_type == 1 && i == 0 && (j > 0 && j < ny_d && k > 0 && k < nz_d))		C1_d[l] = C10_d[nx_d - 1 + offset*j + offset2*k];
		else if (border_type == 1 && i == nx_d && (j > 0 && j < ny_d && k > 0 && k < nz_d))		C1_d[l] = C10_d[1 + offset*j + offset2*k];

		// z
		else if (border_type == 1 && k == 0 && (i > 0 && i < nx_d && j > 0 && j < ny_d))		C1_d[l] = C10_d[i + offset*j + offset2*nz_d - offset2];
		else if (border_type == 1 && k == nz_d && (i > 0 && i < nx_d && j > 0 && j < ny_d))		C1_d[l] = C10_d[i + offset*j + offset2];


		// corner points

		else if (i <= nx_d && j <= ny_d && k <= nz_d && l < n_d) {

			int ii = i + 1 - i / nx_d - ceil(i / (nx_d + 1.0));
			int jj = j + 1 - j / ny_d - ceil(j / (ny_d + 1.0));
			int kk = k + 1 - k / nz_d - ceil(k / (nz_d + 1.0));

			C1_d[l] = C10_d[ii + offset*jj + offset2*kk];
		}


	}
}
__global__ void concentration2(double *ux_d, double *uy_d, double *uz_d, double *vx_d, double *vy_d, double *vz_d, double *T_d, double *T0_d, double *C1_d, double *C10_d, double *C2_d, double *C20_d) {


	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y*blockDim.y;
	unsigned int k = threadIdx.z + blockIdx.z*blockDim.z;
	unsigned int l = i + offset*j + offset2*k;
	if (i <= nx_d && j <= ny_d && k <= nz_d && l < n_d)
	{

		/*
		INNER
		*/
		if (i > 0 && i < nx_d && j > 0 && j < ny_d && k > 0 && k < nz_d)
		{

			//C2
			C2_d[l] = C20_d[l]
				+ tau_d * (
				-vx_d[l] * (C20_d[l + 1] - C20_d[l - 1]) / (2.0 * hx_d)					// (ux_d*d/dx) C2_d
				- vy_d[l] * (C20_d[l + offset] - C20_d[l - offset]) / (2.0 * hy_d)				// (uy_d*d/dy) C2_d
				- vz_d[l] * (C20_d[l + offset2] - C20_d[l - offset2]) / (2.0 * hz_d)				// (uz_d*d/dy) C2_d

				+ ((C20_d[l + 1] - 2.0*C20_d[l] + C20_d[l - 1]) / hx_d / hx_d 	// 1/Pr*(d2/dx2) C2_d
				+ (C20_d[l + offset] - 2.0*C20_d[l] + C20_d[l - offset]) / hy_d / hy_d 	// 1/Pr*(d2/dy2) C2_d
				+ (C20_d[l + offset2] - 2.0*C20_d[l] + C20_d[l - offset2]) / hz_d / hz_d) / Sc22_d	// 1/Pr*(d2/dz2) C2_d

				- (+(T0_d[l + 1] - 2.0*T0_d[l] + T0_d[l - 1]) / hx_d / hx_d 		// 1/Pr*(d2/dx2) T_d
				+ (T0_d[l + offset] - 2.0*T0_d[l] + T0_d[l - offset]) / hy_d / hy_d 	// 1/Pr*(d2/dy2) T_d
				+ (T0_d[l + offset2] - 2.0*T0_d[l] + T0_d[l - offset2]) / hz_d / hz_d) / Sc22_d * psi2_d	// 1/Pr*(d2/dz2) T_d
				);

		}



		/*
		UP-DOWN
		*/

		// y
		else if (j == 0 && (i > 0 && i < nx_d && k > 0 && k < nz_d)) 		C2_d[l] = (4.0*C20_d[l + offset] - C20_d[l + offset * 2]) / 3.0 - 2.0 / 3.0 * hy_d*psi2_d*Q_d;
		else if (j == ny_d && (i > 0 && i < nx_d && k > 0 && k < nz_d))		C2_d[l] = (4.0*C20_d[l - offset] - C20_d[l - offset * 2]) / 3.0 + 2.0 / 3.0 * hy_d*psi2_d*Q_d;

		/*
		CLOSED
		*/

		// x
		else if (border_type == 0 && i == 0 && (j > 0 && j < ny_d && k > 0 && k < nz_d)) 		C2_d[l] = (4.0*C20_d[l + 1] - C20_d[l + 2]) / 3.0;
		else if (border_type == 0 && i == nx_d && (j > 0 && j < ny_d && k > 0 && k < nz_d))		C2_d[l] = (4.0*C20_d[l - 1] - C20_d[l - 2]) / 3.0;

		// z
		else if (border_type == 0 && k == 0 && (i > 0 && i < nx_d && j > 0 && j < ny_d))		C2_d[l] = (4.0*C20_d[l + offset2] - C20_d[l + offset2 * 2]) / 3.0;
		else if (border_type == 0 && k == nz_d && (i > 0 && i < nx_d && j > 0 && j < ny_d))		C2_d[l] = (4.0*C20_d[l - offset2] - C20_d[l - offset2 * 2]) / 3.0;


		/*
		PERIODIC
		*/

		// x

		else if (border_type == 1 && i == 0 && (j > 0 && j < ny_d && k > 0 && k < nz_d))		C2_d[l] = C20_d[nx_d - 1 + offset*j + offset2*k];
		else if (border_type == 1 && i == nx_d && (j > 0 && j < ny_d && k > 0 && k < nz_d))		C2_d[l] = C20_d[1 + offset*j + offset2*k];

		// z
		else if (border_type == 1 && k == 0 && (i > 0 && i < nx_d && j > 0 && j < ny_d))		C2_d[l] = C20_d[i + offset*j + offset2*nz_d - offset2];
		else if (border_type == 1 && k == nz_d && (i > 0 && i < nx_d && j > 0 && j < ny_d))		C2_d[l] = C20_d[i + offset*j + offset2];


		// corner points

		else if (i <= nx_d && j <= ny_d && k <= nz_d && l < n_d) {

			int ii = i + 1 - i / nx_d - ceil(i / (nx_d + 1.0));
			int jj = j + 1 - j / ny_d - ceil(j / (ny_d + 1.0));
			int kk = k + 1 - k / nz_d - ceil(k / (nz_d + 1.0));

			C2_d[l] = C20_d[ii + offset*jj + offset2*kk];
		}


	}
}



__global__ void velocity_correction(double *ux_d, double *uy_d, double *uz_d, double *vx_d, double *vy_d, double *vz_d, double *p_d) {


	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y*blockDim.y;
	unsigned int k = threadIdx.z + blockIdx.z*blockDim.z;
	unsigned int l = i + offset*j + offset2*k;

	if (border_type == 0 && i <= nx_d && j <= ny_d && k <= nz_d && l < n_d)
	{
		if (i > 0 && i < nx_d && j > 0 && j < ny_d && k > 0 && k < nz_d)
		{
			vx_d[l] = ux_d[l] - tau_d / (2.0 * hx_d)*(p_d[l + 1] - p_d[l - 1]);
			vy_d[l] = uy_d[l] - tau_d / (2.0 * hy_d)*(p_d[l + offset] - p_d[l - offset]);
			vz_d[l] = uz_d[l] - tau_d / (2.0 * hz_d)*(p_d[l + offset2] - p_d[l - offset2]);
		}
		else if (i <= nx_d && j <= ny_d && k <= nz_d && l < n_d) {
			vx_d[l] = 0; vy_d[l] = 0; 	vz_d[l] = 0;
		}

	}


	else if (border_type == 1 && i <= nx_d && j <= ny_d && k <= nz_d && l < n_d)
	{
		if (i > 0 && i < nx_d && j > 0 && j < ny_d && k > 0 && k < nz_d)
		{
			vx_d[l] = ux_d[l] - tau_d / 2.0 / hx_d*(p_d[l + 1] - p_d[l - 1]);
			vy_d[l] = uy_d[l] - tau_d / 2.0 / hy_d*(p_d[l + offset] - p_d[l - offset]);
			vz_d[l] = uz_d[l] - tau_d / 2.0 / hz_d*(p_d[l + offset2] - p_d[l - offset2]);
		}
		else if (j == 0 && (i >= 0 && i <= nx_d && k >= 0 && k <= nz_d))
		{
			vx_d[l] = 0.0; vy_d[l] = 0.0; vz_d[l] = 0.0;
		}
		else if (j == ny_d && (i >= 0 && i <= nx_d && k >= 0 && k <= nz_d))
		{
			vx_d[l] = 0.0; vy_d[l] = 0.0; vz_d[l] = 0.0;
		}
		else if (i == 0 && (j > 0 && j < ny_d && k > 0 && k < nz_d))
		{
			int ll = nx_d - 1 + offset*j + offset2*k;
			vx_d[l] = ux_d[ll] - tau_d / 2.0 / hx_d*(p_d[ll + 1] - p_d[ll - 1]);
			vy_d[l] = uy_d[ll] - tau_d / 2.0 / hy_d*(p_d[ll + offset] - p_d[ll - offset]);
			vz_d[l] = uz_d[ll] - tau_d / 2.0 / hz_d*(p_d[ll + offset2] - p_d[ll - offset2]);
		}
		else if (i == nx_d && (j > 0 && j < ny_d && k > 0 && k < nz_d))
		{
			int ll = 1 + offset*j + offset2*k;
			vx_d[l] = ux_d[ll] - tau_d / 2.0 / hx_d*(p_d[ll + 1] - p_d[ll - 1]);
			vy_d[l] = uy_d[ll] - tau_d / 2.0 / hy_d*(p_d[ll + offset] - p_d[ll - offset]);
			vz_d[l] = uz_d[ll] - tau_d / 2.0 / hz_d*(p_d[ll + offset2] - p_d[ll - offset2]);
		}
		else if (k == 0 && (i > 0 && i < nx_d && j > 0 && j < ny_d))
		{
			int ll = i + offset*j + offset2*nz_d - offset2;
			vx_d[l] = ux_d[ll] - tau_d / 2.0 / hx_d*(p_d[ll + 1] - p_d[ll - 1]);
			vy_d[l] = uy_d[ll] - tau_d / 2.0 / hy_d*(p_d[ll + offset] - p_d[ll - offset]);
			vz_d[l] = uz_d[ll] - tau_d / 2.0 / hz_d*(p_d[ll + offset2] - p_d[ll - offset2]);
		}
		else if (k == nz_d && (i > 0 && i < nx_d && j > 0 && j < ny_d))
		{
			int ll = i + offset*j + offset2;
			vx_d[l] = ux_d[ll] - tau_d / 2.0 / hx_d*(p_d[ll + 1] - p_d[ll - 1]);
			vy_d[l] = uy_d[ll] - tau_d / 2.0 / hy_d*(p_d[ll + offset] - p_d[ll - offset]);
			vz_d[l] = uz_d[ll] - tau_d / 2.0 / hz_d*(p_d[ll + offset2] - p_d[ll - offset2]);
		}
	}

	else if (i <= nx_d && j <= ny_d && k <= nz_d && l < n_d) {
		int ii = i + 1 - i / nx_d - ceil(i / (nx_d + 1.0));
		int jj = j + 1 - j / ny_d - ceil(j / (ny_d + 1.0));
		int kk = k + 1 - k / nz_d - ceil(k / (nz_d + 1.0));

		vx_d[l] = 0;
		vy_d[l] = 0;
		vz_d[l] = 0;

	}

}


__global__ void Poisson(double *ux_d, double *uy_d, double *uz_d, double *p_d, double *p0_d)
{
	//период условие отключено
	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y*blockDim.y;
	unsigned int k = threadIdx.z + blockIdx.z*blockDim.z;
	unsigned int l = i + offset*j + offset2*k;
	//double psiav0 = 0.0; double psiav = 0.0; double eps = 1.0; int k = 0;

	/*

	closed

	*/

	if (border_type == 0 && i <= nx_d && j <= ny_d && k <= nz_d && l < n_d)
	{
		if (i > 0 && i < nx_d && j > 0 && j < ny_d && k > 0 && k < nz_d)
		{
			p_d[l] =
				-(ux_d[l + 1] - ux_d[l - 1]  /*/ tau_d / 2.0 / hx_d */
				+ uy_d[l + offset] - uy_d[l - offset]  /* / tau_d / 2.0 / hy_d */
				+ uz_d[l + offset2] - uz_d[l - offset2]) / tau_d / 2.0 / hz_d;

			if (i == 1) 				p_d[l] += 2.0 / 3.0 / hx_d / hx_d*(p0_d[l + 1] - p0_d[l] - hx_d / tau_d*ux_d[l - 1]);
			else if (i == nx_d - 1) 	p_d[l] += 2.0 / 3.0 / hx_d / hx_d*(p0_d[l - 1] - p0_d[l] + hx_d / tau_d*ux_d[l + 1]);
			else						p_d[l] += 1.0 / hx_d / hx_d*(p0_d[l + 1] + p0_d[l - 1] - 2.0*p0_d[l]);

			if (j == 1)					p_d[l] += 2.0 / 3.0 / hy_d / hy_d*(p0_d[l + offset] - p0_d[l] - hy_d / tau_d*uy_d[l - offset]);
			else if (j == ny_d - 1) 	p_d[l] += 2.0 / 3.0 / hy_d / hy_d*(p0_d[l - offset] - p0_d[l] + hy_d / tau_d*uy_d[l + offset]);
			else						p_d[l] += 1.0 / hy_d / hy_d*(p0_d[l + offset] + p0_d[l - offset] - 2.0*p0_d[l]);

			if (k == 1)					p_d[l] += 2.0 / 3.0 / hz_d / hz_d*(p0_d[l + offset2] - p0_d[l] - hz_d / tau_d*uz_d[l - offset2]);
			else if (k == nz_d - 1)		p_d[l] += 2.0 / 3.0 / hz_d / hz_d*(p0_d[l - offset2] - p0_d[l] + hz_d / tau_d*uz_d[l + offset2]);
			else						p_d[l] += 1.0 / hz_d / hz_d*(p0_d[l + offset2] + p0_d[l - offset2] - 2.0*p0_d[l]);

			p_d[l] *= tau_p_d;
			p_d[l] += p0_d[l];
		}


		else if (j == 0 && (i > 0 && i < nx_d && k > 0 && k < nz_d))		p_d[l] = (4.0*p0_d[l + offset] - p0_d[l + offset * 2]) / 3.0 - uy_d[l] * 2.0 * hy_d / tau_d / 3.0;
		else if (j == ny_d && (i > 0 && i < nx_d && k > 0 && k < nz_d))		p_d[l] = (4.0*p0_d[l - offset] - p0_d[l - offset * 2]) / 3.0 + uy_d[l] * 2.0 * hy_d / tau_d / 3.0;
		else if (i == 0 && (j > 0 && j < ny_d && k > 0 && k < nz_d))		p_d[l] = (4.0*p0_d[l + 1] - p0_d[l + 2]) / 3.0 - ux_d[l] * 2.0 * hx_d / tau_d / 3.0;
		else if (i == nx_d && (j > 0 && j < ny_d && k > 0 && k < nz_d))		p_d[l] = (4.0*p0_d[l - 1] - p0_d[l - 2]) / 3.0 + ux_d[l] * 2.0 * hx_d / tau_d / 3.0;
		else if (k == 0 && (i > 0 && i < nx_d && j > 0 && j < ny_d))		p_d[l] = (4.0*p0_d[l + offset2] - p0_d[l + offset2 * 2]) / 3.0 - uz_d[l] * 2.0 * hz_d / tau_d / 3.0;
		else if (k == nz_d && (i > 0 && i < nx_d && j > 0 && j < ny_d))		p_d[l] = (4.0*p0_d[l - offset2] - p0_d[l - offset2 * 2]) / 3.0 + uz_d[l] * 2.0 * hz_d / tau_d / 3.0;

		else if (i <= nx_d && j <= ny_d && k <= nz_d && l < n_d) {
			int ii = i + 1 - i / nx_d - ceil(i / (nx_d + 1.0));
			int jj = j + 1 - j / ny_d - ceil(j / (ny_d + 1.0));
			int kk = k + 1 - k / nz_d - ceil(k / (nz_d + 1.0));
			p_d[l] = p0_d[ii + offset*jj + offset2*kk];
			//p_d[l] = 0;
			//int ii = 0;
			//int jj = 0;
			//int kk = 0;

			//if (i == 0)	ii = 1;		if (i == nx_d) ii = -1;
			//if (j == 0)	jj = 1;		if (j == ny_d) jj = -1;
			//if (k == 0)	kk = 1;		if (k == nz_d) kk = -1;

			//p_d[l] = (4.0*p0_d[l + 1 * ii + offset*jj + offset2*kk] - p0_d[l + 2 * ii + offset*jj + offset2*kk]) / 3.0
			//	- ii* ux_d[l] * 2.0 / 3.0*hx_d / tau_d
			//	- jj* uy_d[l] * 2.0 / 3.0*hy_d / tau_d
			//	- kk* uz_d[l] * 2.0 / 3.0*hz_d / tau_d;
		}
	}

	/*

	periodic

	*/


	else if (border_type == 1 && i <= nx_d && j <= ny_d && k <= nz_d && l < n_d)
	{
		if (i > 0 && i < nx_d && j > 0 && j < ny_d && k > 0 && k < nz_d)
		{
			p_d[l] =
				-(ux_d[l + 1] - ux_d[l - 1]  /*/ tau_d / 2.0 / hx_d */
				+ uy_d[l + offset] - uy_d[l - offset]  /* / tau_d / 2.0 / hy_d */
				+ uz_d[l + offset2] - uz_d[l - offset2]) / tau_d / 2.0 / hz_d;

			/*if (i == 1) 				p_d[l] += 2.0 / 3.0 / hx_d / hx_d*(p0_d[l + 1] - p0_d[l] - hx_d / tau_d*ux_d[l - 1]);
			else if (i == nx_d - 1) 	p_d[l] += 2.0 / 3.0 / hx_d / hx_d*(p0_d[l - 1] - p0_d[l] + hx_d / tau_d*ux_d[l + 1]);
			else*/						p_d[l] += 1.0 / hx_d / hx_d*(p0_d[l + 1] + p0_d[l - 1] - 2.0*p0_d[l]);

			if (j == 1)					p_d[l] += 2.0 / 3.0 / hy_d / hy_d*(p0_d[l + offset] - p0_d[l] - hy_d / tau_d*uy_d[l - offset]);
			else if (j == ny_d - 1) 	p_d[l] += 2.0 / 3.0 / hy_d / hy_d*(p0_d[l - offset] - p0_d[l] + hy_d / tau_d*uy_d[l + offset]);
			else						p_d[l] += 1.0 / hy_d / hy_d*(p0_d[l + offset] + p0_d[l - offset] - 2.0*p0_d[l]);

			/*if (k == 1)					p_d[l] += 2.0 / 3.0 / hz_d / hz_d*(p0_d[l + offset2] - p0_d[l] - hz_d / tau_d*uz_d[l - offset2]);
			//else if (k == nz_d - 1)		p_d[l] += 2.0 / 3.0 / hz_d / hz_d*(p0_d[l - offset2] - p0_d[l] + hz_d / tau_d*uz_d[l + offset2]);
			*/							p_d[l] += 1.0 / hz_d / hz_d*(p0_d[l + offset2] + p0_d[l - offset2] - 2.0*p0_d[l]);

			p_d[l] *= tau_p_d;
			p_d[l] += p0_d[l];
		}

		else if (j == 0 && (i > 0 && i < nx_d && k >= 0 && k <= nz_d))		p_d[l] = (4.0*p0_d[l + offset] - p0_d[l + offset * 2]) / 3.0 - uy_d[l] * 2.0 / 3.0*hy_d / tau_d;
		else if (j == ny_d && (i > 0 && i < nx_d && k >= 0 && k <= nz_d))	p_d[l] = (4.0*p0_d[l - offset] - p0_d[l - offset * 2]) / 3.0 + uy_d[l] * 2.0 / 3.0*hy_d / tau_d;
		else if (i == 0 && (j > 0 && j < ny_d && k > 0 && k < nz_d))		p_d[l] = p0_d[nx_d - 1 + offset*j + offset2*k];
		else if (i == nx_d && (j > 0 && j < ny_d && k > 0 && k < nz_d))		p_d[l] = p0_d[1 + offset*j + offset2*k];
		else if (k == 0 && (i > 0 && i < nx_d && j > 0 && j < ny_d))		p_d[l] = p0_d[i + offset*j + offset2*nz_d - offset2];
		else if (k == nz_d && (i > 0 && i < nx_d && j > 0 && j < ny_d))		p_d[l] = p0_d[i + offset*j + offset2];

		else if (i <= nx_d && j <= ny_d && k <= nz_d && l < n_d) {

			int ii = i + 1 - i / nx_d - ceil(i / (nx_d + 1.0));
			int jj = j + 1 - j / ny_d - ceil(j / (ny_d + 1.0));
			int kk = k + 1 - k / nz_d - ceil(k / (nz_d + 1.0));
			p_d[l] = p0_d[ii + offset*jj + offset2*kk];
			//int ii = 0;
			//int jj = 0;
			//int kk = 0;

			//if (i == 0)	ii = 1;		if (i == nx_d) ii = -1;
			//if (j == 0)	jj = 1;		if (j == ny_d) jj = -1;
			//if (k == 0)	kk = 1;		if (k == nz_d) kk = -1;

			//p_d[l] = (4.0*p0_d[l + 1 * ii + offset*jj + offset2*kk] - p0_d[l + 2 * ii + offset*jj + offset2*kk]) / 3.0
			//	- ii* ux_d[l] * 2.0 / 3.0*hx_d / tau_d
			//	- jj* uy_d[l] * 2.0 / 3.0*hy_d / tau_d
			//	- kk* uz_d[l] * 2.0 / 3.0*hz_d / tau_d;
		}
	}

}

__global__ void reduction00(double *data, unsigned int n, double* reduced) {
	extern __shared__ double shared[];


	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;

	shared[tid] = (i < n) ? abs(data[i]) : 0;

	if (i + blockDim.x < n)	shared[tid] += abs(data[i + blockDim.x]);


	__syncthreads();


	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s>32; s >>= 1)
	{
		if (tid < s)
		{
			shared[tid] += shared[tid + s];
		}

		__syncthreads();
	}


	if (tid < 32)
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockDim.x >= 64) shared[tid] += shared[tid + 32];
		// Reduce final warp using shuffle
		for (int offset = warpSize / 2; offset > 0; offset /= 2)
		{
			shared[tid] += __shfl_down((float)shared[tid], offset);
		}
	}



	if (tid == 0) {
		reduced[blockIdx.x] = shared[0];
	}


}
__global__ void reduction0(double *data, unsigned int n, double* reduced) {
	extern __shared__ double shared[];


	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < n) {
		shared[tid] = abs(data[i]);
	}
	else
	{
		shared[tid] = 0.0;
	}

	__syncthreads();


	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (tid < s)
		{
			shared[tid] += shared[tid + s];
		}

		__syncthreads();
	}

	if (tid == 0) {
		reduced[blockIdx.x] = shared[0];
	}


}
__global__ void reduction(double *data, unsigned int n, double* reduced) {
	extern  __shared__  double shared[];

	unsigned int tid = threadIdx.x;
	//unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		shared[tid] = abs(data[i]);
		//if (i + blockDim.x  < n) shared[tid] += abs(data[i + blockDim.x]);
	}
	else
	{
		shared[tid] = 0.0;
	}

	__syncthreads();

	if (blockDim.x >= 1024) {
		if (tid < 512) { shared[tid] += shared[tid + 512]; } __syncthreads();
	}
	if (blockDim.x >= 512) {
		if (tid < 256) { shared[tid] += shared[tid + 256]; } __syncthreads();
	}
	if (blockDim.x >= 256) {
		if (tid < 128) { shared[tid] += shared[tid + 128]; } __syncthreads();
	}
	if (blockDim.x >= 128) {
		if (tid < 64) { shared[tid] += shared[tid + 64]; } __syncthreads();
	}
	if (tid < 32)
	{
		if (blockDim.x >= 64) shared[tid] += shared[tid + 32];
		if (blockDim.x >= 32) shared[tid] += shared[tid + 16];
		if (blockDim.x >= 16) shared[tid] += shared[tid + 8];
		if (blockDim.x >= 8) shared[tid] += shared[tid + 4];
		if (blockDim.x >= 4) shared[tid] += shared[tid + 2];
		if (blockDim.x >= 2) shared[tid] += shared[tid + 1];
	}




	if (tid == 0) {
		reduced[blockIdx.x] = shared[0];
		//if (blockDim.x==1) *last = shared[0];
	}


}


__global__ void swap_one(double* f_old, double* f_new) {
	unsigned int l = blockIdx.x*blockDim.x + threadIdx.x;
	if (l < n_d)	f_old[l] = f_new[l];
}
__global__ void swap_6(double* f1_old, double* f1_new, double* f2_old, double* f2_new, double* f3_old, double* f3_new, double* f4_old, double* f4_new, double* f5_old, double* f5_new, double* f6_old, double* f6_new) {
	unsigned int l = blockIdx.x*blockDim.x + threadIdx.x;
	if (l < n_d)
	{
		f1_old[l] = f1_new[l];
		f2_old[l] = f2_new[l];
		f3_old[l] = f3_new[l];
		f4_old[l] = f4_new[l];
		f5_old[l] = f5_new[l];
		f6_old[l] = f6_new[l];
	}
}



__global__ void pressure_norm1(double* T_d) {
	dp = T_d[nx_d / 2 + ny_d / 2 * offset + nz_d / 2 * offset2];
	//dp = - psi2_d*1.0 + T_d[nx_d / 2 + nz_d / 2 * offset2];
}
__global__ void pressure_norm2(double* T_d) {
	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y*blockDim.y;
	unsigned int k = threadIdx.z + blockIdx.z*blockDim.z;
	unsigned int l = i + offset*j + offset2*k;

	if (l < n_d)	{
		T_d[l] = T_d[l] - dp;
	}
}



int main() {

	int devID = 0;
	cudaSetDevice(devID);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, devID);
	printf("\nDevice %d: \"%s\"\n", devID, deviceProp.name);

#ifdef __linux__
	system("mkdir -p fields/");
#endif

#ifdef _WIN32
	CreateDirectoryA("fields", NULL);
#endif

	int np;
	double timer1, timer2;

	double pi = 3.1415926535897932384626433832795;
	double *T0_d, *T_d, *p_d, *p0_d, *ux_d, *uy_d, *uz_d, *vx_d, *vy_d, *vz_d, *C1_d, *C10_d, *C2_d, *C20_d;  //_d - device (GPU) 
	double *T0_h, *T_h, *p_h, *p0_h, *ux_h, *uy_h, *uz_h, *vx_h, *vy_h, *vz_h, *C1_h, *C10_h, *C2_h, *C20_h;	//_h - host (CPU)
	double *psiav_array, *psiav_array_h, *psiav_d, *psiav_h, psiav0_h, eps_h, eps0;		 //  temporal variables
	double hx, hy, hz, Lx, Ly, Lz, tau, tau_p, m, dRa, Ra, Pr, psiav, psiav0, eps, psi1, psi2, psiS, Sc11, Sc12, Sc21, Sc22, Q, alpha, sinA_h, cosA_h; //parameters 
	double Ek, Ek_old, Vmax;
	unsigned int nx, ny, nz, iter, niter, nout, nxout, nyout, off, off2, k, mx, my, mz, border, tt;					  //parameters
	double Vxm, Vym, Vzm, pm, Tm, Cm, pmin, Tmin, Cmin;
	double	x1, y1, z1, xx1, yy1, zz1, x2, y2, z2, xx2, yy2, zz2, x3, y3, z3, xx3, yy3, zz3;
	double *test_d, *test_h, *test2_d, *test2_h;
	double AMx, AMy, AMz, AMabs;
	//ofstream maxvals("MAX.txt");
	//maxvals << "#max i j k" << endl;

	np = 1;


	Q = -1.0;
	Sc11 = 100.0;
	Sc12 = 0;
	Sc21 = 0;
	Sc22 = 1000.0;
	Pr = 10.0;



	psi1 = 0.0;
	psiS = -0.2;
	psi2 = psiS - psi1;

	dRa = -100;
	Ra = 5000 - dRa;
	tau = (1e-4); //tau /= 2;
	tt = round(1.0 / tau);
	ny = +30;
	nx = +30;
	nz = +30;
	hy = 1.0 / 30;
	hx = hy;
	hz = hy;

	Ly = ny * hy;
	Lx = nx * hx;
	Lz = nz * hz;

	pi = 3.1415926535897932384626433832795;
	alpha = 0.0;
	cosA_h = cos(alpha*pi / 180);
	sinA_h = sin(alpha*pi / 180);

	//hy = 1.0 / ny;
	//nx = (unsigned int)Lx*ny;
	//hx = Lx / nx;

	tau_p = 0.1*hx*hx;
	eps0 = 1e-5;


	//0 is for the closed cavity, 1 is for the periodic one
	border = 0;

	bool read_fields = true;



	m = 0.0; psiav = 0.0; iter = 0; Ek = 0; Ek_old = 0;

	//offsets variables are used in multi-dimensional arrays which are built as a single-dimension array 
	//for example: f[5][10][15] = f[5 + off * 10 + off2 * 15]
	off /*offset*/ = nx + 1;
	off2 /*offset2*/ = (nx + 1)* (ny + 1);

	unsigned int size_l = (nx + 1) * (ny + 1) * (nz + 1); //Number of all nodes/elements 
	cout << "size_l=" << size_l << endl;
	if (size_l <= 1024 || size_l >= 1024 * 1024 * 1024) { cout << "data is too small or too large" << endl; return 0; }
	size_t size_b /*size (in) bytes*/ = size_l * sizeof(double); //sizeof(double) = 8 bytes

	size_t thread_x_d /*the dimension of x in a block*/ = 8;
	size_t thread_y_d /*the dimension of y in a block*/ = 8;
	size_t thread_z_d /*the dimension of z in a block*/ = 8;
	size_t threads_per_block = thread_x_d * thread_y_d * thread_z_d;

	dim3 gridD(ceil((nx + 1.0) / thread_x_d), ceil((ny + 1.0) / thread_y_d), ceil((nz + 1.0) / thread_z_d)); //неидеально
	dim3 blockD(thread_x_d, thread_y_d, thread_z_d);
	cout << "gridD.x=" << gridD.x << endl;
	cout << "gridD.y=" << gridD.y << endl;
	cout << "gridD.z=" << gridD.z << endl;


	//setting for the reduction procedure 
		unsigned long long int *Gp, *Np;
		unsigned int s = 0, thread_p = 1024;
		

		unsigned int GN = size_l;
		while (true)
		{
			s++;
			GN = ceil(GN / (thread_p + 0.0));
			if (GN == 1)  break;
		}
		GN = size_l;
		std::cout << "the number of reduction = " << s << endl;
		Gp = new unsigned long long int[s];
		Np = new unsigned long long int[s];
		for (int i = 0; i < s; i++)
			Gp[i] = GN = ceil(GN / (thread_p + 0.0));
		Np[0] = size_l;
		for (int i = 1; i < s; i++)
			Np[i] = Gp[i - 1];
		int last_reduce = pow(2, ceil(log2(Np[s - 1] + 0.0))); //last_reduce = pow(2, ceil(log2(size_l / 1024)));
		std::cout << "last reduction = " << last_reduce << endl;
		(s != 1) ? std::cout << "sub array for the Poisson solver = " << Np[1] << endl :
			std::cout << "it shouldn't be here" << endl;
		double *arr[10];

	//allocating memory for arrays on CPU
	T0_h = (double*)malloc(size_b); 	T_h = (double*)malloc(size_b);
	p_h = (double*)malloc(size_b);		p0_h = (double*)malloc(size_b);
	ux_h = (double*)malloc(size_b);		uy_h = (double*)malloc(size_b);		uz_h = (double*)malloc(size_b);
	vx_h = (double*)malloc(size_b);		vy_h = (double*)malloc(size_b);		vz_h = (double*)malloc(size_b);
	C1_h = (double*)malloc(size_b);    C10_h = (double*)malloc(size_b);
	C2_h = (double*)malloc(size_b);    C20_h = (double*)malloc(size_b);
	psiav_h = (double*)malloc(sizeof(double)); 	psiav_array_h = (double*)malloc(size_b / threads_per_block);
	test_h = (double*)malloc(size_b); 	test2_h = (double*)malloc(size_b);
	for (int l = 0; l < size_l; l++) { T0_h[l] = 0.0; T_h[l] = 0.0; p_h[l] = 0.0; p0_h[l] = 0.0; ux_h[l] = 0.0; uy_h[l] = 0.0; vx_h[l] = 0.0; vy_h[l] = 0.0; }

	//allocating memory for arrays on GPU
	cudaMalloc((void**)&T0_d, size_b); 	cudaMalloc((void**)&T_d, size_b);
	cudaMalloc((void**)&p_d, size_b); 	cudaMalloc((void**)&p0_d, size_b);
	cudaMalloc((void**)&ux_d, size_b);	cudaMalloc((void**)&uy_d, size_b); 	cudaMalloc((void**)&uz_d, size_b);
	cudaMalloc((void**)&vx_d, size_b);	cudaMalloc((void**)&vy_d, size_b); 	cudaMalloc((void**)&vz_d, size_b);
	cudaMalloc((void**)&C1_d, size_b);  cudaMalloc((void**)&C10_d, size_b);
	cudaMalloc((void**)&C2_d, size_b);  cudaMalloc((void**)&C20_d, size_b);
	(s != 1) ? cudaMalloc((void**)&psiav_array, sizeof(double)*Np[1]) : cudaMalloc((void**)&psiav_array, sizeof(double));
	//cudaMalloc((void**)&psiav_array, size_b / threads_per_block); 	
	cudaMalloc((void**)&psiav_d, sizeof(double));
	
		arr[0] = p_d;
		for (int i = 1; i <= s; i++)
			arr[i] = psiav_array;

	//ofstream is a class to write data in a file 
	ofstream Ra_tab;
	ofstream integrals;
	ofstream k_number;
	ofstream particle1, particle2, particle3;





	ifstream read("all.txt");
	read.good() == true ? cout << "a file is opened to continue" << endl : cout << " a file is not found, I'l do it from the start" << endl;
	if (read.good() == false) read_fields = false;


	if (read_fields == false) {
		Ra_tab.open("Ra.dat");
		integrals.open("integrals.dat", std::ofstream::app);
		Ra_tab << "Ra, Ek, Vmax, Vc, time(min), t, AMx, AMy, AMz, AMabs" << endl;



		//начальные условия
		for (int i = 0; i <= nx; i++) {
			for (int j = 0; j <= ny; j++) {
				for (int k = 0; k <= nz; k++) {
					//T_h[i + off*j + off2*k] = 0.0;
					//T_h[i + off*j + off2*k] = 1.0 - hy*j;
					//if (i == 0 || i == nx || j == 0 || j == ny || k == 0 || k == nz)
					T_h[i + off*j + off2*k] = 1.0 - hy*j;
					C1_h[i + off*j + off2*k] = T_h[i + off*j + off2*k] * psi1;
					C2_h[i + off*j + off2*k] = T_h[i + off*j + off2*k] * psi2;
					vx_h[i + off*j + off2*k] = 0.0;
					vy_h[i + off*j + off2*k] = 0.0;
					vz_h[i + off*j + off2*k] = 0.0;
					p_h[i + off*j + off2*k] = 0.0;
				}
			}
		}

		vy_h[nx / 2 + off * ny / 2 + off2 * nz / 2] = 0.1;

	}
	else
	{
		Ra_tab.open("Ra.dat", std::ofstream::app);
		integrals.open("integrals.dat", std::ofstream::app);


		string str;
		string substr;
		stringstream ss;
		getline(read, str); getline(read, str);
		ss << str; ss >> substr; ss >> substr; iter = atoi(substr.c_str()); //time
		ss >> substr; ss >> substr; Ra = atof(substr.c_str()); Ra -= dRa; //Ra
		for (int i = 0; i <= nx; i++) {
			for (int j = 0; j <= ny; j++) {
				for (int k = 0; k <= nz; k++) {
					ss.str(""); ss.clear(); getline(read, str); ss << str;
					ss >> substr; ss >> substr; ss >> substr; //skip reading x,y,z
					ss >> substr; vx_h[i + off*j + off2*k] = atof(substr.c_str());
					ss >> substr; vy_h[i + off*j + off2*k] = atof(substr.c_str());
					ss >> substr; vz_h[i + off*j + off2*k] = atof(substr.c_str());
					ss >> substr; C1_h[i + off*j + off2*k] = atof(substr.c_str());
					ss >> substr; C2_h[i + off*j + off2*k] = atof(substr.c_str());
					ss >> substr; T_h[i + off*j + off2*k] = atof(substr.c_str());
					ss >> substr; p_h[i + off*j + off2*k] = atof(substr.c_str());
				}
			}
		}
		cout << "continue from t= " << iter*tau << endl;
	}
	read.close();



	
	cudaMemcpy(T0_d, T_h, size_b, cudaMemcpyHostToDevice); 	cudaMemcpy(T_d, T_h, size_b, cudaMemcpyHostToDevice);
	cudaMemcpy(p0_d, p_h, size_b, cudaMemcpyHostToDevice); 	cudaMemcpy(p_d, p_h, size_b, cudaMemcpyHostToDevice);
	cudaMemcpy(ux_d, vx_h, size_b, cudaMemcpyHostToDevice); 	cudaMemcpy(uy_d, vy_h, size_b, cudaMemcpyHostToDevice); 	cudaMemcpy(uz_d, vz_h, size_b, cudaMemcpyHostToDevice);
	cudaMemcpy(vx_d, vx_h, size_b, cudaMemcpyHostToDevice); 	cudaMemcpy(vy_d, vy_h, size_b, cudaMemcpyHostToDevice); 	cudaMemcpy(vz_d, vz_h, size_b, cudaMemcpyHostToDevice);
	cudaMemcpy(C1_d, C1_h, size_b, cudaMemcpyHostToDevice); 	cudaMemcpy(C10_d, C1_h, size_b, cudaMemcpyHostToDevice);
	cudaMemcpy(C2_d, C2_h, size_b, cudaMemcpyHostToDevice); 	cudaMemcpy(C20_d, C2_h, size_b, cudaMemcpyHostToDevice);


syda_go: // if we reach a state that is considered to be stationary we go to this go-to-mark
	Ra += dRa;



	//copying some constant parameters to the fast constant memory
	cudaMemcpyToSymbol(hx_d, &hx, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(hy_d, &hy, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(hz_d, &hz, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Lx_d, &Lx, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Ly_d, &Ly, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Lz_d, &Lz, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(nx_d, &nx, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(ny_d, &ny, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(nz_d, &nz, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(n_d, &size_l, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Ra_d, &Ra, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Pr_d, &Pr, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(tau_d, &tau, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(tau_p_d, &tau_p, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Sc11_d, &Sc11, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Sc12_d, &Sc12, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Sc21_d, &Sc21, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Sc22_d, &Sc22, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(psi1_d, &psi1, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(psi2_d, &psi2, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(psiS_d, &psiS, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Q_d, &Q, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(offset, &off, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(offset2, &off2, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(sinA, &sinA_h, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(cosA, &cosA_h, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(border_type, &border, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
	//cout << iter*tau << " Ek= " << Ek(nx, ny, nz, Lx, Lz, vx_h, vy_h, vz_h) << endl;




	//just printing parameters from GPU to be confident they are passed correctly 
	hello << <1, 1 >> > ();
	cudaDeviceSynchronize();


	// we use a function from OpenMP standart to measure real time of calculating 
	timer1 = clock()/CLOCKS_PER_SEC;

	//just time in the double precision format
	double timeq = 0.0;



	// the main time loop of the whole calculation procedure
	while (true) {


		iter = iter + 1; 	timeq = timeq + tau;



		//1st step, calculating of time evolutionary parts of velocity (quasi-velocity) , temperature and concentration 
		{
			//quasi_velocity_and_temperature_and_concentration << < gridD, blockD >> > (ux_d, uy_d, uz_d, vx_d, vy_d, vz_d, T_d, T0_d, C1_d, C10_d, C2_d, C20_d);

			quasi_velocity << < gridD, blockD >> > (ux_d, uy_d, uz_d, vx_d, vy_d, vz_d, T_d, T0_d, C1_d, C10_d, C2_d, C20_d);
			temperature << < gridD, blockD >> > (ux_d, uy_d, uz_d, vx_d, vy_d, vz_d, T_d, T0_d, C1_d, C10_d, C2_d, C20_d);

			//concentration1 << < gridD, blockD >> > (ux_d, uy_d, uz_d, vx_d, vy_d, vz_d, T_d, T0_d, C1_d, C10_d, C2_d, C20_d);
			concentration2 << < gridD, blockD >> > (ux_d, uy_d, uz_d, vx_d, vy_d, vz_d, T_d, T0_d, C1_d, C10_d, C2_d, C20_d);
		}


		//2nd step, Poisson equation for pressure 
		{
			eps = 1.0; 		psiav0 = 0.0;		psiav = 0.0;		k = 0;

			while (eps > eps0*psiav0 || k < 2)
			{

				psiav = 0.0;
				k++;

				Poisson << < gridD, blockD >> > (ux_d, uy_d, uz_d, p_d, p0_d);
				//if (timeq > 10) {	swap_one << < ceil(size_l / 1024.0), 1024 >> > (p0_d, p_d); break;}



				for (int i = 0; i < s; i++)
					reduction0 << < Gp[i], 1024, 1024 * sizeof(double) >> > (arr[i], Np[i], arr[i + 1]);




				swap_one << < ceil(size_l / 1024.0), 1024 >> > (p0_d, p_d);
				cudaMemcpy(&psiav, psiav_array, sizeof(double), cudaMemcpyDeviceToHost);

				eps = abs(psiav - psiav0); 
				psiav0 = psiav;


				
				//cout << setprecision(30) << "GPU sum=" << psiav << endl << endl;
				//pause


				if (k % 1000 == 0) {
					cudaMemcpy(p_h, p_d, size_b, cudaMemcpyDeviceToHost);
					cout << k << "  " << setprecision(15) << p_h[1 + off + off2] - p_h[nx - 1 + off*(ny - 1) + off2*(nz - 1)] << " " << eps << endl;
					//cout << "p_iter= " << k << endl;
				}

			}

		}
		




		//3
		velocity_correction << < gridD, blockD >> > (ux_d, uy_d, uz_d, vx_d, vy_d, vz_d, p_d);
		swap_6 << < ceil(size_l / 1024.0), 1024 >> >(ux_d, vx_d, uy_d, vy_d, uz_d, vz_d, T0_d, T_d, C10_d, C1_d, C20_d, C2_d);




		//4
		if (iter % (tt / 5) == 0 || iter == 1) {

			cout << setprecision(15) << endl;
			cout << fixed << endl;
			cudaMemcpy(vx_h, vx_d, size_b, cudaMemcpyDeviceToHost);
			cudaMemcpy(vy_h, vy_d, size_b, cudaMemcpyDeviceToHost);
			cudaMemcpy(vz_h, vz_d, size_b, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_h, p_d, size_b, cudaMemcpyDeviceToHost);
			cudaMemcpy(T_h, T_d, size_b, cudaMemcpyDeviceToHost);
			cudaMemcpy(C2_h, C2_d, size_b, cudaMemcpyDeviceToHost);

			velocity(nx, ny, nz, Lx, Lz, vx_h, vy_h, vz_h, Ek, Vmax);

			//cout << iter*tau << endl;
			cout << endl;
			cout << "Vx=" << maxval(vx_h, size_l) << endl;
			cout << "Vy=" << maxval(vy_h, size_l) << endl;
			cout << "Vz=" << maxval(vz_h, size_l) << endl;
			cout << "Vy_c=" << vy_h[10 + off * 10 + off2 * 10] << endl;
			cout << "p=" << maxval(p_h, size_l) << endl;
			Tm = maxval(T_h, size_l);
			Cm = maxval(C2_h, size_l);
			pm = maxval(p_h, size_l);
			Tmin = minval(T_h, size_l);
			Cmin = minval(C2_h, size_l);
			pmin = minval(p_h, size_l);
			angular_momentum(vx_h, vy_h, vz_h, nx, ny, nz, hx, hy, hz, AMx, AMy, AMz, AMabs);
			cout << "T=" << Tm << endl;
			cout << "C=" << Cm << endl;
			cout << "p_iter=" << k << endl;
			cout << "Ek= " << Ek << endl;
			cout << "dEk= " << abs(Ek - Ek_old) << endl;
			cout << "t=" << iter*tau << endl;
			timer


			if (iter == 1)	integrals << "t, Ek, Vmax,  time(min), Vc,  dEk, Tm, Cm, pm, C2*, p*, dPtd, dT, dC, dP, Tc, dTv, dCv, AMx, AMy, AMz, AMabs" << endl;
			integrals << setprecision(20) << fixed;
			integrals << tau*iter << " " << Ek << " " << Vmax << " " << (timer2 - timer1) / 60 << " " << vy_h[nx / 2 + off * ny / 2 + off2 * nz / 2] << " " << abs(Ek - Ek_old) << " " <<
				Tm << " " << Cm << " " << pm << " " << C2_h[10 + off * ny / 2 + off2 *(nz - 10)] << " " << p_h[3 + off * 3 + off2 *(nz - 20)]
				<< " " << p_h[nx / 2 + off * 0 + off2 *(nz / 2)] - p_h[nx / 2 + off *ny + off2 *(nz / 2)]
				<< " " << Tm - Tmin << " " << Cm - Cmin << " " << pm - pmin << " " << T_h[nx / 2 + off * ny / 2 + off2 * nz / 2]
				<< " " << max_vertical_difference(T_h, nx, ny, nz)
				<< " " << max_vertical_difference(C2_h, nx, ny, nz)
				<< " " << AMx << " " << AMy << " " << AMz << " " << AMabs
				<< endl;

			//printf("%30.25f \n", Ek); pause



			if ((iter*tau > 1500 && abs(Ek - Ek_old)  < 1e-8 && Ra == 5000) || (iter*tau > 1000 && abs(Ek - Ek_old) < 1e-7 && Ra < 5000)) {
				Ra_tab << Ra << " " << Ek << " " << Vmax << " " << vy_h[nx / 2 + off * ny / 2 + off2 * nz / 2] << " " << (timer2 - timer1) / 60 << " " << tau*iter 
					<< " " << AMx << " " << AMy << " " << AMz << " " << AMabs
					<< endl;
				write_fields(nx, ny, nz, hx, hy, hz, off, off2, size_b, iter, tau, Ra,
					vx_h, vy_h, vz_h, T_h, C1_h, C2_h, p_h,
					vx_d, vy_d, vz_d, T_d, C1_d, C2_d, p_d);
				iter = 0;  goto syda_go;
			}

			Ek_old = Ek;
		}



		if (iter % (tt * 50) == 0 && timeq > 10)
			backup(nx, ny, nz, hx, hy, hz, off, off2, size_b, iter, tau, Ra,
				vx_h, vy_h, vz_h, T_h, C1_h, C2_h, p_h,
				vx_d, vy_d, vz_d, T_d, C1_d, C2_d, p_d);


		if (iter % (tt * 10) == 0 || iter == 1) 
			write_fields(nx, ny, nz, hx, hy, hz, off, off2, size_b, iter, tau, Ra,
				vx_h, vy_h, vz_h, T_h, C1_h, C2_h, p_h,
				vx_d, vy_d, vz_d, T_d, C1_d, C2_d, p_d);






		// the end of 4




	} //the end of the main loop



	return 0;
}









