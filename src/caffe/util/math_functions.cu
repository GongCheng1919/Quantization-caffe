#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>

#include <curand_kernel.h> //curand device function 

#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"

namespace caffe {

template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float* X,
                           cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double* X,
                            cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_scale<int>(const int n, const int alpha, const int *x,
                            int* y) {
}

template <>
void caffe_gpu_scale<unsigned>(const int n, const unsigned alpha, const unsigned *x,
                             unsigned* y) {
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}


template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void log_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template <>
void caffe_gpu_log<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_log<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <typename Dtype>
__global__ void sqrt_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = sqrt(a[index]);
  }
}

template <>
void caffe_gpu_sqrt<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_sqrt<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
                                      - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}
//实现gpu求和同步
template <typename Dtype>
__global__ void ternary_kernel(const int n, const Dtype delta, const Dtype* X, Dtype* Y){
  CUDA_KERNEL_LOOP(index, n) {
	const Dtype a = X[index];
    Y[index] = (a>delta) - (a<-delta);
  }
}

template <>
void caffe_gpu_ternary<int>(const int N, const int delta, const int* X, int* Y,int* alpha) {
// NOT IMPLEMENT
}

template <>
void caffe_gpu_ternary<unsigned>(const int N, const unsigned delta, const unsigned* X, unsigned* Y,unsigned* alpha) {
// NOT IMPLEMENT
}

template <>
void caffe_gpu_ternary<float>(const int N, const float delta, const float* X, float* Y,float* alpha) {
// NOT IMPLEMENT
	//clock_t ed;
	//double stime;
	//ed=clock();
  ternary_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, delta, X, Y);
  if(X!=Y){
	  const float* ternary_data=(const float*)Y;
	  float pa = 0;
	  caffe_gpu_dot(N, ternary_data, X, &pa); 
	  float pb = 0;
	  caffe_gpu_dot(N, ternary_data, ternary_data, &pb);
	  *alpha = (pa) / ((pb) + 1e-6);
  }else{
	  *alpha=1;//如果X，Y是同一数据，则无法做缩放值的计算，只能返回1
   }
  //stime=(double)(clock() - ed)/CLOCKS_PER_SEC;
//LOG(INFO)<<"quantize time is "<<stime*1000<<"ms";
  //std::cout<<"pa="<<pa<<", pb="<<pb<<", alpha="<<*alpha<<std::endl;
}

template <>
void caffe_gpu_ternary<double>(const int N, const double delta, const double* X, double* Y,double* alpha) {
// NOT IMPLEMENT
  ternary_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, delta, X, Y);
  if(X!=Y){
	  const double* ternary_data=(const double*)Y;
	  double pa = 0;
	  caffe_gpu_dot(N, ternary_data, X, &pa); 
	  double pb = 0;
	  caffe_gpu_dot(N, ternary_data, ternary_data, &pb);
	  *alpha = (pa) / ((pb) + 1e-6);
  }else{
	  *alpha=1;//如果X，Y是同一数据，则无法做缩放值的计算，只能返回1
  }
  //std::cout<<"pa="<<pa<<", pb="<<pb<<", alpha="<<*alpha<<std::endl;
}
template <typename Dtype>
__global__ void quantize_kernel(const int n,const Dtype fixed_value, const Dtype max_num, const Dtype min_num, const Dtype* X, Dtype* Y){
	CUDA_KERNEL_LOOP(index, n){
		Dtype x_q = X[index] / fixed_value;
		// create a rand N satisfy uniform(-0.5,0.5)
		//x_q+=Y[index];//给x_q加上随机数，此处的Y已经初始化为随机数了
		x_q = (x_q > 0.0) ? floor(x_q + 0.5) : ceil(x_q - 0.5);
		if (x_q >= max_num)
		{
			Y[index] = max_num - 1;
		}else if (x_q <= min_num){
			Y[index] = min_num + 1;
		}else{
			Y[index] = x_q;
		}
	}
}

//
template<typename Dtype>
void caffe_gpu_getcudavalue(const Dtype* X,const int index, Dtype* result){
	CUDA_CHECK(cudaMemcpy(result, X+index, sizeof(Dtype), cudaMemcpyDeviceToHost));
}
template<>
void caffe_gpu_quantizea<int>(const int count, const int* X, int* Y, int* fixed_point, int max_bits, bool calc_fixed_point){}
template<>
void caffe_gpu_quantizea<unsigned>(const int count, const unsigned* X, unsigned* Y, int* fixed_point, int max_bits, bool calc_fixed_point){}
template<>
void caffe_gpu_quantizea<double>(const int count, const double* X, double* Y, int* fixed_point, int max_bits, bool calc_fixed_point){
	//寻找最大值的定点位置
	if(calc_fixed_point){
		int max_index;
		CUBLAS_CHECK(cublasIdamax(Caffe::cublas_handle(),count, X, 1, &max_index));
		double max_n;
		caffe_gpu_getcudavalue(X,max_index-1,&max_n);//cublas的数组下标从1开始
		max_n=fabs(max_n);
		if (max_n == 0){
			*fixed_point = 0;
		}else{
			//计算定点位置
			*fixed_point = floor(log2(max_n)) - (max_bits - 2);
		}
	}
	const double max_num = pow(2, max_bits-1);
	const double min_num = -max_num;
	const double fixed_value = pow(2, *fixed_point);
	//先将Y初始化为-0.5到0.5的随机数,然后使用Y对X的量化结构加噪声
	//caffe_gpu_rng_uniform(count,double(-0.5),double(0.5),Y);
	//计算量化结果
	quantize_kernel << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(count, fixed_value,max_num,min_num, X, Y);
}
template<>
void caffe_gpu_quantizea<float>(const int count, const float* X, float* Y, int* fixed_point, int max_bits, bool calc_fixed_point){
	//寻找最大值的定点位置
	//clock_t ed;
	//double stime;
	//ed=clock();
	if(calc_fixed_point){
		int max_index;
		CUBLAS_CHECK(cublasIsamax(Caffe::cublas_handle(),count, X, 1, &max_index));
		float max_n;
		caffe_gpu_getcudavalue(X,max_index-1,&max_n);//cublas的数组下标从1开始
		//std::cout<<"max_n="<<max_n;
		//std::cout<<"before max value is "<<max_n << " in index : "<<max_index<<std::endl;
		max_n=fabsf(max_n);
	//std::cout<<", abs max_n="<<max_n<<std::endl;
		if (max_n == 0){
			*fixed_point = 0;
		}else{
			//计算定点位置
			*fixed_point = floor(log2(max_n)) - (max_bits - 2);
		}
	}
	//stime=(double)(clock() - ed)/CLOCKS_PER_SEC;
	//LOG(INFO)<<"find max time is "<<stime*1000<<"ms";
	const float max_num = pow(2, max_bits-1);
	const float min_num = -max_num;
	const float fixed_value = pow(2, *fixed_point);
	//std::cout<<"fixed_point is "<<*fixed_point<<std::endl;
	//先将Y初始化为-0.5到0.5的随机数,然后使用Y对X的量化结构加噪声
	//ed=clock();
	//caffe_gpu_rng_uniform(count,float(-0.5),float(0.5),Y);
	//stime=(double)(clock() - ed)/CLOCKS_PER_SEC;
	//LOG(INFO)<<"generate rand time is "<<stime*1000<<"ms";
	//计算量化结果
	//ed=clock();
	quantize_kernel << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(count, fixed_value,max_num,min_num, X, Y);
	//stime=(double)(clock() - ed)/CLOCKS_PER_SEC;
	//LOG(INFO)<<"quantize time is "<<stime*1000<<"ms";
}
//Clip quantization
template <typename Dtype>
__global__ void quantize_kernel(const int n,Dtype* X, Dtype max_pos,Dtype max_neg){
	CUDA_KERNEL_LOOP(index, n){
		if(X[index]>max_pos){
			X[index]=max_pos;
		}
		if(X[index]<max_neg){
			X[index]=max_neg;
		}
	}
}
template<>
void caffe_gpu_quantizeb<int>(const int count, int* X, int max_bits){}
template<>
void caffe_gpu_quantizeb<unsigned>(const int count, unsigned* X, int max_bits){}
template<>
void caffe_gpu_quantizeb<float>(const int count, float* X, int max_bits){
	//实现剪切
	float max_pos=(float)pow(2,max_bits-1)-1;
	float max_neg=-(float)pow(2,max_bits-1);
	quantize_kernel << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(count,X,max_pos,max_neg);
}
template<>
void caffe_gpu_quantizeb<double>(const int count, double* X, int max_bits){
	//实现剪切
	double max_pos=(double)pow(2,max_bits-1)-1;
	double max_neg=-(double)pow(2,max_bits-1);
	quantize_kernel << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(count,X,max_pos,max_neg);
}
//scale clip
template <typename Dtype>
__global__ void round_kernel(const int n,Dtype* X){
	CUDA_KERNEL_LOOP(index, n){
		X[index] = (X[index] > 0.0) ? floor(X[index] + 0.5) : ceil(X[index] - 0.5);
	}
}
template<>
void caffe_gpu_quantizec<int>(const int count, const int* X, int* Y,int* alpha, int max_bits){}
template<>
void caffe_gpu_quantizec<unsigned>(const int count, const unsigned* X, unsigned* Y,unsigned* alpha, int max_bits){}
template<>
void caffe_gpu_quantizec<float>(const int count, const float* X, float* Y,float* alpha, int max_bits){
	//首先找到最大值，进行缩放，缩放到[-127,127]之间（-128不能取到）
	int max_index;
	CUBLAS_CHECK(cublasIsamax(Caffe::cublas_handle(),count, X, 1, &max_index));
	float max_n;
	caffe_gpu_getcudavalue(X,max_index-1,&max_n);//cublas的数组下标从1开始
	max_n=fabsf(max_n);
	//scale:scaler=127/max_n;
	float dst_range=(float)pow(2,max_bits-1)-1;
	float scaler=dst_range/max_n;
	*alpha=1/scaler;
	caffe_gpu_scale(count, scaler, X, Y);
	////量化为整型
	round_kernel << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(count,Y);
}
template<>
void caffe_gpu_quantizec<double>(const int count, const double* X, double* Y,double* alpha, int max_bits){
	//首先找到最大值，进行缩放，缩放到[-127,127]之间（-128不能取到）
	int max_index;
	CUBLAS_CHECK(cublasIdamax(Caffe::cublas_handle(),count, X, 1, &max_index));
	double max_n;
	caffe_gpu_getcudavalue(X,max_index-1,&max_n);//cublas的数组下标从1开始
	max_n=fabsf(max_n);
	//scale:scaler=127/max_n;
	double dst_range=(double)pow(2,max_bits-1)-1;
	double scaler=dst_range/max_n;
	*alpha=1/scaler;
	caffe_gpu_scale(count, scaler, X, Y);
	////量化为整型
	round_kernel << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(count,Y);
}
//meanstd
__global__ void block_Ssum(const int n, const float *X, float *result)
{
	extern __shared__ float sdata[];
	sdata[threadIdx.x] = 0;
	//__syncthreads();//等待所有线程把自己负责的元素载入到共享内存
	CUDA_KERNEL_LOOP(i, n){
		sdata[threadIdx.x] += X[i];
	}
	__syncthreads();//等待所有线程把自己负责的元素载入到共享内存
		for(int offset = blockDim.x / 2;offset > 0;offset >>= 1){
			if(threadIdx.x < offset)//控制只有某些线程才进行操作。
			{
				// add a partial sum upstream to our own
				sdata[threadIdx.x] += sdata[threadIdx.x + offset];
			}
			// wait until all threads in the block have
			// updated their partial sums
			__syncthreads();
		}
		if(threadIdx.x == 0)
		{
			result[blockIdx.x] = sdata[0];
		}
}
__global__ void block_Dsum(const int n, const double *X, double *result)
{
	extern __shared__ double sdata1[];
	sdata1[threadIdx.x] = 0;
	__syncthreads();//等待所有线程把自己负责的元素载入到共享内存
	CUDA_KERNEL_LOOP(i, n){
		sdata1[threadIdx.x] += X[i];
	}
	__syncthreads();//等待所有线程把自己负责的元素载入到共享内存
		for(int offset = blockDim.x / 2;offset > 0;offset >>= 1){
			if(threadIdx.x < offset)//控制只有某些线程才进行操作。
			{
				// add a partial sum upstream to our own
				sdata1[threadIdx.x] += sdata1[threadIdx.x + offset];
			}
			// wait until all threads in the block have
			// updated their partial sums
			__syncthreads();
		}
		if(threadIdx.x == 0)
		{
			result[blockIdx.x] = sdata1[0];
		}
}
template<class DType>
__global__ void block_sum(const DType *input,DType *per_block_results,const size_t n)
{
	extern __shared__ DType sdata[];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	// load input into __shared__ memory //一个线程负责把一个元素从全局内存载入到共享内存
	DType x = 0;
	if(i < n)
	{
		x = input[i];
	}
	sdata[threadIdx.x] = x;
	__syncthreads();//等待所有线程把自己负责的元素载入到共享内存
	// contiguous range pattern//块内进行合并操作，每次合并变为一半.注意threadIdx.x是块内的偏移，上面算出的i是全局的偏移。
	for(int offset = blockDim.x / 2;offset > 0;offset >>= 1){
		if(threadIdx.x < offset)//控制只有某些线程才进行操作。
		{
			// add a partial sum upstream to our own
			sdata[threadIdx.x] += sdata[threadIdx.x + offset];
		}
		// wait until all threads in the block have
		// updated their partial sums
		__syncthreads();
	}
	// thread 0 writes the final result//每个块的线程0负责存放块内求和的结果
	if(threadIdx.x == 0)
	{
		per_block_results[blockIdx.x] = sdata[0];
	}
}

template <>
void caffe_gpu_sum<int>(const int n, const int* x, int* y){}
template <>
void caffe_gpu_sum<unsigned>(const int n, const unsigned* x, unsigned* y){}
template <>
void caffe_gpu_sum<float>(const int n, const float* x, float* y){
	const int block_size = CAFFE_CUDA_NUM_THREADS;//线程块的大小。
	int num_elements = n;
	int num_blocks = CAFFE_GET_BLOCKS(n);
	float *dev_input_array = 0;
	float *dev_block_sums = 0;//一个线程块一个和。
	while(num_elements > block_size)
	{
		if(dev_block_sums == 0)//第一次
		{
			dev_input_array = (float*)x;
		}
		else //除了第一次
		{
			//除了第一次的中间结果都进行销毁
			if(dev_input_array != x){
				CUDA_CHECK(cudaFree(dev_input_array));
			}
			dev_input_array = dev_block_sums;
		}
		//num_blocks = (num_elements/block_size) + ((num_elements%block_size) ? 1 : 0);
		//给输出结果分配内存
		CUDA_CHECK(cudaMalloc((void**)&dev_block_sums, sizeof(float) * (num_blocks )));
		// launch one kernel to compute, per-block, a partial sum//把每个线程块的和求出来
		block_Ssum<<<num_blocks,block_size, block_size*sizeof(float)>>>(num_elements,dev_input_array, dev_block_sums);
		//每一块有一个结果，总的结果数量就是输出的块数
		num_elements = num_blocks;
		//下次的迭代块数
		num_blocks = CAFFE_GET_BLOCKS(num_blocks);
	}
	float* res=0;
	CUDA_CHECK(cudaMalloc((void**)&res, sizeof(float)));
	float result=0;
	//能够用一个块装下所有的数据了，这时候做一个块的计算
	block_Ssum<<<1,num_elements,num_elements*sizeof(float)>>>(num_elements,dev_block_sums, res );
	CUDA_CHECK(cudaMemcpy(&result, res, sizeof(float), cudaMemcpyDeviceToHost));
	//释放最后一次的结果
	CUDA_CHECK(cudaFree(res));
	CUDA_CHECK(cudaFree(dev_block_sums));
	*y=result;
}
template <>
void caffe_gpu_sum<double>(const int n, const double* x, double* y){
	const int block_size = CAFFE_CUDA_NUM_THREADS;//线程块的大小。
	int num_elements = n;
	int num_blocks = CAFFE_GET_BLOCKS(n);
	double *dev_input_array = 0;
	double *dev_block_sums = 0;//一个线程块一个和。
	while(num_elements > block_size)
	{
		if(dev_block_sums == 0)//第一次
		{
			dev_input_array = (double*)x;
		}
		else //除了第一次
		{
			//除了第一次的中间结果都进行销毁
			if(dev_input_array != x){
				CUDA_CHECK(cudaFree(dev_input_array));
			}
			dev_input_array = dev_block_sums;
		}
		//num_blocks = (num_elements/block_size) + ((num_elements%block_size) ? 1 : 0);
		//给输出结果分配内存
		CUDA_CHECK(cudaMalloc((void**)&dev_block_sums, sizeof(double) * (num_blocks )));
		// launch one kernel to compute, per-block, a partial sum//把每个线程块的和求出来
		block_Dsum<<<num_blocks,block_size,block_size*sizeof(double)>>>(num_elements,dev_input_array, dev_block_sums);
		//每一块有一个结果，总的结果数量就是输出的块数
		num_elements = num_blocks;
		//下次的迭代块数
		num_blocks = CAFFE_GET_BLOCKS(num_blocks);
	}
	double* res=0;
	double result=0;
	CUDA_CHECK(cudaMalloc((void**)&res, sizeof(double)));
	//能够用一个块装下所有的数据了，这时候做一个块的计算
	block_Dsum<<<1,num_elements,num_elements*sizeof(double)>>>(num_elements,dev_block_sums, res );
	CUDA_CHECK(cudaMemcpy(&result, res, sizeof(double), cudaMemcpyDeviceToHost));
	//释放最后一次的结果
	CUDA_CHECK(cudaFree(res));
	//释放最后一次的结果
	CUDA_CHECK(cudaFree(dev_block_sums));
	*y=result;
}

template<>
void caffe_gpu_meanstd<int>(const int count, const int* X, int& mean, int& std){}
template<>
void caffe_gpu_meanstd<unsigned>(const int count, const unsigned* X, unsigned& mean, unsigned& std){}
template<>
void caffe_gpu_meanstd<float>(const int count, const float* X, float& mean, float& stds){
	//calc mean
	float total;
	//CUBLAS_CHECK(cublasSsum(Caffe::cublas_handle(), count, X, 1, &total));
	//float X0=0,Xc=0;
	//caffe_gpu_getcudavalue(X,0, &X0);
	//caffe_gpu_getcudavalue(X,count-1, &Xc);
	//LOG(INFO)<<"count="<<count<<",X[0]="<<X0<<",X[count-1]="<<Xc;
	//LOG(INFO)<<"calc sum by GPU!";
	//caffe_gpu_sum(count,X,&total);
	thrust::device_ptr<const float> dev_ptr(X);
	total=thrust::reduce(dev_ptr, dev_ptr + size_t(count), (float)0, thrust::plus<float>());
	mean=total/count;
	//LOG(INFO)<<"GPU sum is "<<total;
	//calc std
	//LOG(INFO)<<"calc square_sum by GPU!";
	caffe_gpu_dot(count, X, X,&total);
	stds=sqrt(total/count-pow(mean,2));
	//LOG(INFO)<<"GPU mean="<<mean<<",std="<<stds;
}
template<>
void caffe_gpu_meanstd<double>(const int count, const double* X, double& mean, double& stds){
	//calc mean
	double total;
	//CUBLAS_CHECK(cublasDsum(Caffe::cublas_handle(), count, X, 1, &total));
	thrust::device_ptr<const double> dev_ptr(X);
	total=thrust::reduce(dev_ptr, dev_ptr + size_t(count), (double)0, thrust::plus<double>());
	mean=total/count;
	//calc std
	caffe_gpu_dot(count, X, X,&total);
	stds=sqrt(total/count-pow(mean,2));
	//LOG(INFO)<<"GPU mean="<<mean<<",std="<<stds;
}

//ulq
//ULQ：Q=Round(      Clip          ((F-delta)*alpha+0.5))#alpha=1/alpha,此处是为了避免除法
//		       [1-2^(k-1),2^(k-1)]
template <typename Dtype>
__global__ void ulq_kernel(const int n,const Dtype mean_old,const Dtype sigma_old,const Dtype maxs,const Dtype mins,const Dtype* X,Dtype* Y,bool restore){
	CUDA_KERNEL_LOOP(index, n){
		//scale
		Y[index]=(X[index]-mean_old)*sigma_old+0.5;
		//clip
		if(Y[index]>maxs){Y[index]=maxs;}
		if(Y[index]<mins){Y[index]=mins;}
		//round
		Y[index] = (Y[index] > 0.0) ? floor(Y[index] + 0.5) : ceil(Y[index] - 0.5);
		//需要还原，宝贝
		//还原为相同的range，以进行前向传播
		if (restore){
			Y[index]=((Y[index]-0.5)/sigma_old)+mean_old;
		}
	}
}
template<>
void caffe_gpu_ulq<int>(const int count, const int mean_old, const int sigma_old,const int* X, int* Y, int max_bits,bool restore){}
template<>
void caffe_gpu_ulq<unsigned>(const int count, const unsigned mean_old, const unsigned sigma_old,const unsigned* X, unsigned* Y, int max_bits,bool restore){}
template<>
void caffe_gpu_ulq<float>(const int count, const float mean_old, const float sigma_old,const float* X, float* Y, int max_bits,bool restore){
	float mins=1-pow(2,max_bits-1);
	float maxs=pow(2,max_bits-1);
	ulq_kernel << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(count,mean_old,sigma_old,maxs,mins,X,Y,restore);
}
template<>
void caffe_gpu_ulq<double>(const int count, const double mean_old, const double sigma_old,const double* X, double* Y, int max_bits,bool restore){}


}  // namespace caffe
